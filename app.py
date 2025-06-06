from flask import Flask, jsonify, render_template, request, send_file, session, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import unicodedata
import re
import newspaper
import os
from vncorenlp import VnCoreNLP
import csv
import json
import pandas as pd
import io
import uuid
import logging
import numpy as np
import sys
import time
from newspaper import Config
import random
from urllib.parse import urlparse

torch.set_num_threads(1)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", f"dev_secret_{os.urandom(12).hex()}")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO_ID = 'DatNth/PhoBERT-v2-topic-cls'
TOKENIZER_REPO_ID = 'vinai/phobert-base-v2'

DEFAULT_VNCORE_JAR = os.path.join(APP_ROOT, 'vncorenlp_lib', 'VnCoreNLP-1.2.jar')
VNCORE_JAR = os.environ.get('VNCORENLP_JAR_PATH', DEFAULT_VNCORE_JAR)

STOPWORDS_FILE_PATH = os.path.join(APP_ROOT, 'stopwords_tokenized.txt')
TEMP_FILES_ROOT_DIR = "/tmp/app_temp_files"
NO_CHUNK_TOKEN_LIMIT = 256
CHUNK_SIZE = 128
MODEL_MAX_LEN = 256 
DEFAULT_STRATEGY = 'sum_logits'
CHUNK_STRIDE = 0.25

segmenter = None
tokenizer = None
model = None
device = torch.device("cpu")
label_map = []
chunk_processor = None
stopwords_set = set()

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
]

if not app.debug:
    app.logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not app.logger.handlers:
        app.logger.addHandler(handler)

HTML_TAG_REGEX = re.compile(r'<[^>]+>')
URL_REGEX = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
EMAIL_REGEX = re.compile(r'\S+@\S+')
CREDIT_REGEX = re.compile(r'\s*(ảnh|nguồn|video|minh họa|theo|nguồn ảnh|ảnh chụp màn hình|Nguồn tin|Nguồn|Theo)\s*:\s*[\w\s\d\(\)\-\.,\/:]+(\.|$)', flags=re.IGNORECASE | re.UNICODE)
NON_VI_TEXT_REGEX = re.compile(r"[^_a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", flags=re.UNICODE)
MULTI_SPACE_REGEX = re.compile(r'\s+')
EXTRA_CREDITS_PATTERNS = [
    re.compile(r'\s*\(ảnh:\s*[\w\s\d\.\-]+\)$', flags=re.IGNORECASE | re.UNICODE),
    re.compile(r'\s*\([\w\s]*theo\s+[\w\s\d\.\-]+\)$', flags=re.IGNORECASE | re.UNICODE)
]
EMPTY_PUNCT_REGEX = re.compile(r'^[\W_.,?!]+$')


def is_safe_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https') and parsed.hostname \
           and not parsed.hostname.startswith('127.') \
           and 'localhost' not in parsed.hostname \
           and not parsed.hostname.startswith('internal') \
           and not parsed.hostname.startswith('169.254.')

def make_token_chunks(all_token_ids, chunk_size_val, stride_ratio_val):
    if not all_token_ids: return []
    chunks = []
    stride = int(chunk_size_val * stride_ratio_val)
    if stride <= 0 : stride = chunk_size_val // 2
    for i in range(0, len(all_token_ids), stride):
        one_chunk = all_token_ids[i : i + chunk_size_val]
        if one_chunk: chunks.append(one_chunk)
    return chunks

def is_url_string(text_data):
    return re.match(r'^(https?://[^\s]+)$', text_data.strip()) is not None

class PhoBERTChunkClassifier(nn.Module):
    def __init__(self, base_model_obj, tokenizer_obj, max_seq_len=MODEL_MAX_LEN):
        super().__init__()
        self.base_model = base_model_obj
        self.tokenizer = tokenizer_obj
        self.max_len = max_seq_len

    def forward(self, list_of_token_id_chunks):
        num_labels = getattr(self.base_model.config, 'num_labels', 1)
        if not list_of_token_id_chunks:
            return torch.empty(0, num_labels).to(device)
        
        input_ids_list = []
        attention_masks_list = []

        for chunk_ids in list_of_token_id_chunks:
            input_ids = [self.tokenizer.cls_token_id] + chunk_ids + [self.tokenizer.sep_token_id]
            if len(input_ids) > self.max_len:
                input_ids = input_ids[:self.max_len - 1] + [self.tokenizer.sep_token_id]

            attention_mask = [1] * len(input_ids)
            padding_len = self.max_len - len(input_ids)

            input_ids += [self.tokenizer.pad_token_id] * padding_len
            attention_mask += [0] * padding_len

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)

        input_ids_tensor = torch.tensor(input_ids_list, device=device)
        attention_mask_tensor = torch.tensor(attention_masks_list, device=device)

        was_training = self.base_model.training
        if was_training:
            self.base_model.eval()

        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

        if was_training:
            self.base_model.train()

        return outputs.logits


class DocumentPredictionAggregator:
    def __init__(self, strategy='sum_logits'):
        self.selected_strategy = strategy
        VALID_STRATEGIES = ['first_chunk', 'majority_vote', 'mean_probabilities', 'max_probability', 'sum_logits']
        if self.selected_strategy not in VALID_STRATEGIES:
            raise ValueError(f"Chiến lược '{self.selected_strategy}' không hợp lệ.")

    def aggregate(self, all_chunk_logits):
        if not isinstance(all_chunk_logits, torch.Tensor) or all_chunk_logits.size(0) == 0:
            return -1, None
        label_id = -1; probabilities = None
        try:
            if self.selected_strategy == 'first_chunk':
                probabilities = F.softmax(all_chunk_logits[0], dim=0)
                label_id = torch.argmax(probabilities).item()
            elif self.selected_strategy == 'majority_vote':
                chunk_predicted_ids = torch.argmax(all_chunk_logits, dim=1)
                if chunk_predicted_ids.numel() > 0:
                    label_id = torch.mode(chunk_predicted_ids, dim=0)[0].item()
                    winner_indices = (chunk_predicted_ids == label_id).nonzero(as_tuple=True)[0]
                    chosen_logit_idx = winner_indices[0] if len(winner_indices) > 0 else 0
                    probabilities = F.softmax(all_chunk_logits[chosen_logit_idx], dim=0)
                else: return -1, None
            elif self.selected_strategy == 'mean_probabilities':
                chunk_probs = F.softmax(all_chunk_logits, dim=1)
                probabilities = torch.mean(chunk_probs, dim=0)
                label_id = torch.argmax(probabilities).item()
            elif self.selected_strategy == 'max_probability':
                chunk_probs = F.softmax(all_chunk_logits, dim=1)
                max_chunk_probs, _ = torch.max(chunk_probs, dim=1)
                if max_chunk_probs.numel() > 0:
                    best_chunk_idx = torch.argmax(max_chunk_probs)
                    probabilities = chunk_probs[best_chunk_idx]
                    label_id = torch.argmax(probabilities).item()
                else: return -1, None
            elif self.selected_strategy == 'sum_logits':
                total_logits = torch.sum(all_chunk_logits, dim=0)
                probabilities = F.softmax(total_logits, dim=0)
                label_id = torch.argmax(probabilities).item()
        except Exception as e:
            app.logger.error(f"Lỗi tổng hợp chunk: {e}", exc_info=True); return -1, None
        return label_id, probabilities

def load_resources():
    global segmenter, tokenizer, model, device, label_map, chunk_processor, stopwords_set
    app.logger.info("Đang tải tài nguyên...")
    os.makedirs(TEMP_FILES_ROOT_DIR, exist_ok=True)

    try:
        if VNCORE_JAR and os.path.exists(VNCORE_JAR):
            segmenter = VnCoreNLP(VNCORE_JAR, annotators="wseg", max_heap_size='-Xmx1g')
    except Exception as e: app.logger.error(f"Lỗi VnCoreNLP: {e}", exc_info=True); segmenter = None

    if STOPWORDS_FILE_PATH and os.path.exists(STOPWORDS_FILE_PATH):
        try:
            with open(STOPWORDS_FILE_PATH, "r", encoding="utf-8") as f:
                stopwords_set.update([line.strip().lower() for line in f if line.strip()])
        except Exception as e: app.logger.error(f"Lỗi stopwords: {e}.", exc_info=True)
    
    default_labels_list = ["Lỗi_Tải_Nhãn"] 
    label_map = default_labels_list
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO_ID, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO_ID, output_attentions=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); model.eval()
        
        if hasattr(model, 'config') and hasattr(model.config, 'id2label') and model.config.id2label:
            cfg_id2label = model.config.id2label; cfg_num_labels = len(cfg_id2label)
            if cfg_num_labels > 0:
                temp_map = [""] * cfg_num_labels; is_valid_map = True
                for id_key, name_val in cfg_id2label.items():
                    try:
                        idx_val = int(id_key)
                        if 0 <= idx_val < cfg_num_labels: temp_map[idx_val] = str(name_val)
                        else: is_valid_map = False; break
                    except ValueError: is_valid_map = False; break
                if is_valid_map and all(temp_map): label_map = temp_map
                else: label_map = [f"Nhãn_{i}" for i in range(getattr(model.config, 'num_labels', 1))]
            else: label_map = [f"Nhãn_{i}" for i in range(getattr(model.config, 'num_labels', 1))]
        else: label_map = [f"Nhãn_{i}" for i in range(getattr(model.config, 'num_labels', 1))]
        chunk_processor = PhoBERTChunkClassifier(model, tokenizer)
        chunk_processor.to(device); chunk_processor.eval()
    except Exception as e:
        app.logger.error(f"Lỗi tải model: {e}", exc_info=True)
        raise RuntimeError(f"Không thể tải model/tokenizer: {e}") from e
    app.logger.info("Hoàn tất tải tài nguyên.")

def preprocess_text(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip(): return ""
    text_data = unicodedata.normalize('NFC', raw_text)
    text_data = HTML_TAG_REGEX.sub('', text_data); text_data = URL_REGEX.sub('', text_data)
    text_data = EMAIL_REGEX.sub('', text_data); text_data = CREDIT_REGEX.sub(' ', text_data)
    for pattern_re in EXTRA_CREDITS_PATTERNS: text_data = pattern_re.sub('', text_data).strip()
    text_lines = [ln.strip() for ln in text_data.splitlines() if ln.strip() and not EMPTY_PUNCT_REGEX.match(ln.strip())]
    text_data = " ".join(text_lines).lower()
    text_data = MULTI_SPACE_REGEX.sub(' ', text_data).strip()
    if segmenter and text_data:
        try:
            segmented_sents = segmenter.tokenize(text_data)
            text_data = ' '.join(word_tok for sent in segmented_sents for word_tok in sent)
        except Exception as e: app.logger.warning(f"Lỗi tách từ: {e}")
    text_data = NON_VI_TEXT_REGEX.sub(" ", text_data)
    text_data = MULTI_SPACE_REGEX.sub(' ', text_data).strip()
    final_words = [w for w in text_data.split() if not w.isdigit() and w not in stopwords_set]
    return ' '.join(final_words)

def get_top_predictions(probs_tensor, model_cfg_num_labels):
    global label_map
    if not isinstance(probs_tensor, torch.Tensor) or probs_tensor.numel() == 0: return {"thong_bao": "Tensor xác suất không hợp lệ."}
    num_labels_in_probs = probs_tensor.size(-1); current_labels_list = label_map
    if not current_labels_list or current_labels_list == ["Lỗi_Tải_Nhãn"]:
        current_labels_list = [f"Nhãn_{i}" for i in range(model_cfg_num_labels)]
        if num_labels_in_probs != model_cfg_num_labels: current_labels_list = [f"Nhãn_{i}" for i in range(num_labels_in_probs)]
    num_effective_labels = len(current_labels_list)
    if num_labels_in_probs != num_effective_labels:
        num_effective_labels = num_labels_in_probs
        current_labels_list = [f"Nhãn_{i}" for i in range(num_effective_labels)]
    k_val = min(3, num_effective_labels)
    if k_val <= 0: return {"thong_bao": "Số nhãn là 0."}
    try: top_k_probs_vals, top_k_indices_vals = torch.topk(probs_tensor, k=k_val)
    except RuntimeError:
        k_val = min(probs_tensor.numel(), num_effective_labels)
        if k_val == 0 : return {"thong_bao": "Không có xác suất."}
        top_k_probs_vals, top_k_indices_vals = torch.topk(probs_tensor, k=k_val)
    predictions_dict = {}
    indices_py_list, scores_py_list = top_k_indices_vals.cpu().tolist(), top_k_probs_vals.cpu().tolist()
    for item_idx, item_score in zip(indices_py_list, scores_py_list):
        label_name_val = current_labels_list[item_idx] if 0 <= item_idx < len(current_labels_list) else f"IDNhãn_{item_idx}"
        predictions_dict[label_name_val] = round(item_score, 6) 
    return predictions_dict if predictions_dict else {"thong_bao": "Không xác định được chủ đề."}

def predict_directly(text_data):
    global tokenizer, model, device
    if not model or not tokenizer: return {"loi_model": "Hệ thống chưa sẵn sàng."}
    is_training_state = model.training
    if is_training_state: model.eval()
    try:
        model_inputs = tokenizer(text_data, return_tensors="pt", truncation=True, padding="max_length", max_length=MODEL_MAX_LEN)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        with torch.no_grad(): outputs_data = model(**model_inputs)
        logits_data = outputs_data.logits
        if logits_data.shape[0] == 0: return {"thong_bao": "Model không trả về logits."}
        probs_data = F.softmax(logits_data[0], dim=-1)
        if probs_data.numel() == 0: return {"thong_bao": "Không tính được xác suất."}
        num_labels_from_cfg = getattr(getattr(model, 'config', None), 'num_labels', 1)
        return get_top_predictions(probs_data, num_labels_from_cfg)
    except Exception as e:
        app.logger.error(f"Lỗi dự đoán trực tiếp: {e}", exc_info=True)
        return {"loi_model": "Lỗi dự đoán trực tiếp."}
    finally:
        if is_training_state: model.train()

def predict_with_chunks(text_data, strategy_name):
    global tokenizer, chunk_processor, model
    if not model or not tokenizer or not chunk_processor: return {"loi_model": "Hệ thống chưa sẵn sàng."}
    is_training_state = chunk_processor.base_model.training
    if is_training_state : chunk_processor.eval()
    aggregator_obj = DocumentPredictionAggregator(strategy=strategy_name)
    try:
        all_token_ids = tokenizer.encode(text_data, add_special_tokens=False, truncation=False)
        if strategy_name == "first_chunk":
            text_token_chunks = [all_token_ids[:CHUNK_SIZE]]
        else:
            text_token_chunks = make_token_chunks(all_token_ids, CHUNK_SIZE, CHUNK_STRIDE)
        if not text_token_chunks and all_token_ids: text_token_chunks.append(all_token_ids[:CHUNK_SIZE]) 
        if not text_token_chunks: return {"thong_bao": "Văn bản không thể chia đoạn."}
        all_logits_output = chunk_processor(text_token_chunks)
        if all_logits_output.size(0) == 0: return {"thong_bao": "Không đoạn nào được xử lý."}
        _, final_probs = aggregator_obj.aggregate(all_logits_output)
        if final_probs is None: return {"thong_bao": "Không tổng hợp được kết quả."}
        num_labels_from_cfg = getattr(getattr(model, 'config', None), 'num_labels', 1)
        return get_top_predictions(final_probs, num_labels_from_cfg)
    except Exception as e:
        app.logger.error(f"Lỗi dự đoán theo đoạn: {e}", exc_info=True)
        return {"loi_model": "Lỗi dự đoán theo đoạn."}
    finally:
        if is_training_state: chunk_processor.train()

def get_doc_predictions(text_to_process, selected_strategy):
    if not model or not tokenizer: return {"loi": "Hệ thống chưa sẵn sàng."}
    if not text_to_process or not text_to_process.strip(): return {"thong_bao": "Văn bản đầu vào rỗng."}
    try:
        if selected_strategy == "no_chunking": return predict_directly(text_to_process)
        all_token_ids = tokenizer.encode(text_to_process, add_special_tokens=False, truncation=False)
        num_tokens_in_doc = len(all_token_ids)
        if num_tokens_in_doc <= NO_CHUNK_TOKEN_LIMIT:
            direct_prediction_result = predict_directly(text_to_process)
            if not (direct_prediction_result.get("loi_model") or direct_prediction_result.get("thong_bao")):
                return direct_prediction_result
            if num_tokens_in_doc > CHUNK_SIZE : app.logger.info(f"Thử chunking cho văn bản ngắn ({num_tokens_in_doc} tokens).")
            else: return direct_prediction_result 
        return predict_with_chunks(text_to_process, selected_strategy)
    except Exception as e:
        app.logger.error(f"Lỗi lấy dự đoán: {e}", exc_info=True)
        return {"loi_model": "Lỗi chung khi phân loại."}

def get_suggestions():
    suggestions_file_path = os.path.join(APP_ROOT, "sample_suggestions.json")
    if os.path.exists(suggestions_file_path):
        try:
            with open(suggestions_file_path, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: app.logger.error(f"Lỗi tải gợi ý: {e}", exc_info=True)
    return [
        {"title": "Công nghệ AI và ứng dụng", "description": "Trí tuệ nhân tạo đang ngày càng phổ biến."}, 
        {"title": "Giá xăng tăng, dầu giảm", "url": "https://dantri.com.vn/kinh-doanh/gia-xang-tang-dau-giam-20250529142603419.htm"}, 
        {"title": "Gen Z lười vận động dễ \"làm bạn\" với tiểu đường", "url": "https://dantri.com.vn/nhip-song-tre/gen-z-luoi-van-dong-de-lam-ban-voi-tieu-duong-20221130221428818.htm"}
    ]

def find_df_text_columns(df_input):
    title_keywords = ['title', 'tiêu đề', 'tựa đề', 'header', 'subject', 'name']
    text_keywords = ['text', 'content', 'nội dung', 'văn bản', 'body', 'article', 'full_text', 'abstract', 'description', 'noidung']
    normalized_cols = {col.lower().replace(" ", "").replace("_", ""): col for col in df_input.columns}
    title_col = next((normalized_cols[kw.replace(" ", "").replace("_", "")] for kw in title_keywords if kw.replace(" ", "").replace("_", "") in normalized_cols), None)
    text_col = next((normalized_cols[kw.replace(" ", "").replace("_", "")] for kw in text_keywords if kw.replace(" ", "").replace("_", "") in normalized_cols), None)
    if not text_col and len(df_input.columns) == 1: text_col = df_input.columns[0]
    elif title_col and not text_col: text_col = next((c for c in df_input.columns if c != title_col), df_input.columns[0] if df_input.columns.size > 0 else None)
    elif not title_col and not text_col and df_input.columns.size >= 1:
        text_col = df_input.columns[0]
        if df_input.columns.size >= 2:
            c1n, c2n = df_input.columns[0].lower().replace(" ","").replace("_",""), df_input.columns[1].lower().replace(" ","").replace("_","")
            if any(kw in c1n for kw in title_keywords) and any(kw in c2n for kw in text_keywords): title_col, text_col = df_input.columns[0], df_input.columns[1]
            elif any(kw in c2n for kw in title_keywords) and any(kw in c1n for kw in text_keywords): title_col, text_col = df_input.columns[1], df_input.columns[0]
    return title_col, text_col

# app.py (Chỉ phần sửa đổi hàm get_entry_content)

def get_entry_content(original_text, original_title=""):
    text_for_model, title_out, content_out, error_out = "", original_title or "", original_text or "", None
    
    if is_url_string(original_text or ""):
        url_from_entry = (original_text or "").strip()
        
        crawled_page_title, crawled_text_for_model, crawl_error = fetch_url_content(url_from_entry)
        
        title_out = crawled_page_title
        error_out = crawl_error

        if crawl_error:
            text_for_model = crawled_text_for_model
            content_out = ""
        else:
            text_for_model = crawled_text_for_model
            if crawled_page_title and crawled_text_for_model.startswith(crawled_page_title):
                temp_content = crawled_text_for_model[len(crawled_page_title):].strip()
                if temp_content.startswith('.'):
                    temp_content = temp_content[1:].strip()
                content_out = temp_content
            else:
                content_out = crawled_text_for_model

    else:
        content_out = original_text # original_text chính là nội dung
        text_for_model = f"{original_title}. {original_text}" if original_title else original_text 
        
    return text_for_model.strip(), title_out.strip(), content_out.strip(), error_out
def extract_df_rows(df_in, entries_out):
    title_col, text_col = find_df_text_columns(df_in)
    if text_col:
        for _, row in df_in.iterrows():
            text = str(row.get(text_col, "")) if pd.notna(row.get(text_col)) else ""
            title = str(row.get(title_col, "")) if title_col and pd.notna(row.get(title_col)) else ""
            if text.strip(): entries_out.append({"original_title": title, "original_text": text})
        return True
    return False

def read_file_extract_entries(file_obj):
    name = file_obj.filename; entries = []
    if name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_obj, engine='openpyxl')
        if not extract_df_rows(df, entries):
            df_no_head = pd.read_excel(file_obj, engine='openpyxl', header=None)
            for col in df_no_head.columns:
                for txt in df_no_head[col].dropna().astype(str):
                    if txt.strip(): entries.append({"original_title": "", "original_text": txt})
    elif name.endswith('.csv'):
        raw_bytes = file_obj.read()
        try: str_content = raw_bytes.decode('utf-8-sig')
        except UnicodeDecodeError: str_content = raw_bytes.decode('latin-1')
        try:
            df = pd.read_csv(io.StringIO(str_content))
            if not extract_df_rows(df, entries): raise ValueError("Pandas failed for CSV.")
        except:
            reader = csv.reader(io.StringIO(str_content).readlines())
            header = next(reader, None); title_idx, text_idx = -1, -1
            if header:
                h_lower = [str(h).lower().strip() for h in header]
                for i, h_val in enumerate(h_lower):
                    if any(kw in h_val for kw in ['title', 'tiêu đề']): title_idx = i
                    if any(kw in h_val for kw in ['text', 'content', 'nội dung']): text_idx = i
            for row_items in reader:
                txt = row_items[text_idx] if text_idx != -1 and len(row_items) > text_idx else ""
                title = row_items[title_idx] if title_idx != -1 and len(row_items) > title_idx else ""
                if not txt and row_items: txt = " ".join(filter(None, row_items))
                if txt.strip(): entries.append({"original_title": title, "original_text": txt})
    elif name.endswith('.json'):
        data = json.loads(file_obj.read().decode('utf-8'))
        def parse_json_data(item):
            if isinstance(item, dict):
                title = str(item.get('title', item.get('tiêu đề', '')))
                text = str(item.get('text', item.get('content', item.get('description', ''))))
                if text.strip(): entries.append({"original_title": title, "original_text": text})
                else: 
                    for v_item in item.values(): parse_json_data(v_item)
            elif isinstance(item, list): 
                for sub_item in item: parse_json_data(sub_item)
            elif isinstance(item, str) and item.strip(): entries.append({"original_title": "", "original_text": item})
        parse_json_data(data)
    elif name.endswith('.txt'):
        content = file_obj.read().decode('utf-8')
        docs = [d.strip() for d in re.split(r'\n\s*\n{2,}', content) if d.strip()]
        if not docs and content.strip(): docs = [content.strip()]
        for doc_item in docs: entries.append({"original_title": "", "original_text": doc_item})
    return entries

def predict_entries(entries_list, strategy):
    preview_entries, excel_data = [], []
    for entry in entries_list:
        text_for_pred, title_disp, content_disp, err_msg = get_entry_content(entry["original_text"], entry["original_title"])
        preview_text = entry["original_text"] if is_url_string(entry["original_text"]) else (content_disp[:100] + "..." if len(content_disp) > 100 else content_disp)
        excel_row = {'title': title_disp, 'content': content_disp, 'predicted_label': "N/A", 'error_info': ""}
        if err_msg or not text_for_pred.strip():
            msg = err_msg or 'Văn bản rỗng.'; preview_entries.append({'original_text': preview_text, 'predicted_label': msg})
            excel_row['predicted_label'] = "Lỗi"; excel_row['error_info'] = msg
        else:
            cleaned_text = preprocess_text(text_for_pred)
            if not cleaned_text.strip():
                msg = 'Văn bản rỗng sau tiền xử lý.'; preview_entries.append({'original_text': preview_text, 'predicted_label': msg})
                excel_row['predicted_label'] = "Lỗi"; excel_row['error_info'] = msg
            else:
                preds = get_doc_predictions(cleaned_text, strategy)
                if preds and not any(k in preds for k in ["loi", "thong_bao", "error", "loi_model"]):
                    top_label = max(preds, key=preds.get)
                    preview_entries.append({'original_text': preview_text, 'predicted_label': top_label})
                    excel_row['predicted_label'] = top_label
                else:
                    err = preds.get("loi") or preds.get("thong_bao") or preds.get("error") or preds.get("loi_model") or "Lỗi Dự Đoán"
                    preview_entries.append({'original_text': preview_text, 'predicted_label': err})
                    excel_row['predicted_label'] = "Lỗi"; excel_row['error_info'] = err
        excel_data.append(excel_row)
    return preview_entries, excel_data

def write_results_to_excel(excel_data_output):
    df_out = pd.DataFrame(excel_data_output); buffer = io.BytesIO()
    df_out.to_excel(buffer, index=False, engine='openpyxl'); buffer.seek(0)
    filename = f"batch_results_{uuid.uuid4().hex}.xlsx"
    full_path = os.path.join(TEMP_FILES_ROOT_DIR, filename)
    with open(full_path, "wb") as f: f.write(buffer.getvalue())
    return full_path

def process_uploaded_file(file_obj, strategy):
    try:
        text_entries = read_file_extract_entries(file_obj)
        if not text_entries: return [], None, "File không có nội dung hợp lệ."
        preview_entries, excel_data = predict_entries(text_entries, strategy)
        excel_path = write_results_to_excel(excel_data)
        return preview_entries[:20], excel_path, None
    except Exception as e:
        app.logger.error(f"Lỗi xử lý file '{file_obj.filename}': {e}", exc_info=True)
        return [], None, f"Lỗi xử lý file: {e}"

@app.route('/', methods=['GET'])
def home():
    for key in list(session.keys()):
        if key.startswith('url_report_paths_') or key.startswith('batch_excel_temp_file_'):
            session.pop(key, None)
    return render_template('index.html', 
                           results=None, 
                           article_title="Chưa có nội dung", 
                           num_tokens=0, 
                           current_strategy=DEFAULT_STRATEGY,
                           text_input_value="", 
                           text_title_value="", 
                           url_input_value="",
                           current_input_mode="url", 
                           label_map=label_map,
                           file_results_preview=None, 
                           download_filename_id=None, 
                           temporary_report_id=None, 
                           suggested_articles=get_suggestions(),
                           DEFAULT_CHUNK_STRATEGY=DEFAULT_STRATEGY,
                           NO_CHUNK_TOKEN_LIMIT=NO_CHUNK_TOKEN_LIMIT,
                           CHUNK_SIZE=CHUNK_SIZE,
                           MODEL_MAX_LEN=MODEL_MAX_LEN,
                           text_for_attention_input="")

def fetch_url_content(url):
    page_title, text_for_model, error = "Lỗi URL", "", None
    try:
        cfg = Config();
        cfg.browser_user_agent = random.choice(USER_AGENTS);
        cfg.request_timeout=15
        if not is_safe_url(url):
            return "URL không hợp lệ hoặc nguy hiểm", 400
        article = newspaper.Article(url.strip(), language='vi', fetch_images=False, config=cfg)
        article.download();
        article.parse()
        ext_title, ext_text = (article.title or "").strip(), (article.text or "").strip()
        page_title = ext_title if ext_title else f"URL: {url.strip()}"
        if not ext_text and not ext_title: error = f"Không trích xuất được gì từ URL: {url.strip()}"
        elif not ext_text: error = f"Không trích xuất được nội dung (chỉ có tiêu đề)."
        text_for_model = f"{ext_title}. {ext_text}" if ext_title and ext_text else (ext_text or ext_title or f"URL không hợp lệ.")
    except Exception as e:
        error = f"Lỗi URL: {e}"; page_title = f"Lỗi URL: {url.strip()}"; text_for_model = page_title
        app.logger.error(f"Lỗi URL '{url.strip()}': {e}", exc_info=True)
    return page_title.strip(), text_for_model.strip(), error

def prepare_text_input(main_text, title_text):
    page_title, text_for_model, error = "Không có tiêu đề", "", None
    clean_title, clean_main = (title_text or "").strip(), (main_text or "").strip()
    if not clean_main and not clean_title: error = "Vui lòng nhập tiêu đề hoặc nội dung."
    elif clean_title: page_title = clean_title; text_for_model = f"{clean_title}. {clean_main}" if clean_main else clean_title
    elif clean_main: page_title = clean_main[:100] + ("..." if len(clean_main)>100 else ""); text_for_model = clean_main
    return page_title.strip(), text_for_model.strip(), error

def classify_and_get_results(text_to_classify, current_strategy, view_vars, mode, source_id=""):
    processing_start_time = time.time()
    view_vars["temporary_report_id"] = None
    if not text_to_classify or not text_to_classify.strip():
        view_vars["results"] = {"loi": "Nội dung đầu vào rỗng."}
        view_vars["text_for_attention_input"] = ""
        return False, ""
    
    if len(text_to_classify) > 5000 or any(char in text_to_classify for char in ['<', '>', '{', '}']):
        return "Nội dung không hợp lệ hoặc chứa ký tự không hợp lệ", 400
    
    view_vars["text_for_attention_input"] = text_to_classify 
    
    cleaned_text = preprocess_text(text_to_classify)
    token_count = len(tokenizer.encode(cleaned_text, add_special_tokens=False)) if tokenizer and cleaned_text.strip() else 0
    view_vars["num_tokens"] = token_count
    view_vars["processed_text_preview"] = cleaned_text[:1000] + ("..." if len(cleaned_text) > 1000 else "")

    if not cleaned_text.strip():
        view_vars["results"] = {"loi": "Văn bản rỗng sau tiền xử lý."}
        return False, ""
            
    predictions = get_doc_predictions(cleaned_text, current_strategy)
    view_vars["results"] = predictions

    if predictions and not any(k in predictions for k in ["loi", "thong_bao", "error", "loi_model"]):
        if mode in ["url", "text"]:
            report_id = str(uuid.uuid4().hex)
            report_base = f"report_data_{report_id}"
            report_title = view_vars.get('article_title', source_id)
            
            contents = {
                "detail": f"Nguồn: {report_title}\nNội dung đã xử lý ({len(cleaned_text)} chars, {token_count} tokens):\n{cleaned_text}\n\nDự đoán:\n" + "\n".join([f"- {lbl}: {scr*100:.2f}%" for lbl, scr in predictions.items()]),
                "original": text_to_classify, "processed": cleaned_text
            }
            session_dl_data = {"display_title_for_filename": report_title}
            try:
                for fmt_key, content_data in contents.items():
                    file_path_temp = os.path.join(TEMP_FILES_ROOT_DIR, f"{report_base}_{fmt_key}.txt")
                    with open(file_path_temp, "w", encoding="utf-8") as f: f.write(content_data)
                    session_dl_data[f"{fmt_key}_path"] = file_path_temp
                session[f'url_report_paths_{report_id}'] = session_dl_data
                view_vars["temporary_report_id"] = report_id
            except Exception as e: app.logger.error(f"Lỗi lưu file báo cáo tạm: {e}", exc_info=True)
    
    processing_end_time = time.time()
    classification_time_ms = (processing_end_time - processing_start_time) * 1000
    view_vars["classification_time_ms"] = classification_time_ms
    return True, cleaned_text


@app.route('/classify', methods=['POST'])
def classify_route():
    request_start_time = time.time() 

    form = request.form
    text_val, title_val, url_val = form.get("text", ""), form.get("text_title", ""), form.get("url", "")
    file_obj = request.files.get('file_input')
    strategy_val, input_mode = form.get("chunk_strategy", DEFAULT_STRATEGY), form.get("active_input_mode", "url")
    
    content_for_model, page_title = "", "Không có tiêu đề"; error_msg = None
    file_preview, batch_download_key = None, None
    
    for k_clear in [k for k in session if k.startswith('url_report_paths_') or k.startswith('batch_excel_temp_file_')]:
        session.pop(k_clear, None)

    if input_mode == "url":
        if url_val and is_url_string(url_val):
            page_title, content_for_model, error_msg = fetch_url_content(url_val)
        else: error_msg = "URL không hợp lệ."
    elif input_mode == "text":
        page_title, content_for_model, error_msg = prepare_text_input(text_val, title_val)
    elif input_mode == "file":
        if file_obj and file_obj.filename:
            page_title = f"File: {file_obj.filename}"
            file_preview, temp_excel_path_val, error_msg = process_uploaded_file(file_obj, strategy_val)
            if not error_msg and temp_excel_path_val:
                batch_download_key = f"batch_excel_temp_file_{os.path.basename(temp_excel_path_val)}"
                session[batch_download_key] = temp_excel_path_val 
        else: error_msg = "Vui lòng chọn một file."
    else: error_msg = "Chế độ đầu vào không hợp lệ."

    view_data = {
        "results": None, "article_title": page_title, "processed_text_preview": "", 
        "num_tokens": 0, "current_strategy": strategy_val, "suggested_articles": get_suggestions(),
        "text_input_value": text_val, "text_title_value": title_val, "url_input_value": url_val, 
        "current_input_mode": input_mode, "label_map": label_map, 
        "file_results_preview": file_preview, "download_filename_id": batch_download_key, 
        "temporary_report_id": None, 
        "text_for_attention_input": "",
        "DEFAULT_CHUNK_STRATEGY": DEFAULT_STRATEGY,
        "NO_CHUNK_TOKEN_LIMIT": NO_CHUNK_TOKEN_LIMIT,
        "CHUNK_SIZE": CHUNK_SIZE,
        "MODEL_MAX_LEN": MODEL_MAX_LEN,
        "processing_time_total_ms": 0,
        "classification_time_ms": 0
    }

    if error_msg: 
        view_data["results"] = {"loi": error_msg}
    elif input_mode != "file":
        original_source_id = url_val if input_mode == "url" else page_title
        classify_and_get_results(content_for_model, strategy_val, view_data, input_mode, original_source_id)
    
    request_end_time = time.time()
    processing_time_total_ms = (request_end_time - request_start_time) * 1000
    view_data["processing_time_total_ms"] = processing_time_total_ms
    app.logger.info(f"Tổng thời gian xử lý backend cho /classify: {processing_time_total_ms:.2f} ms")
    if "classification_time_ms" in view_data and view_data["classification_time_ms"] > 0 :
         app.logger.info(f"Thời gian xử lý logic phân loại (classify_and_get_results): {view_data['classification_time_ms']:.2f} ms")
    else:
        view_data["classification_time_ms"] = processing_time_total_ms


    return render_template("index.html", **view_data)


@app.route('/download_url_report/<report_guid>', methods=['GET'])
def download_url_report(report_guid):
    if not report_guid: return "Thiếu ID báo cáo.", 400
    paths_session_key = f'url_report_paths_{report_guid}'
    report_file_paths = session.get(paths_session_key)
    if not report_file_paths: return "Không có dữ liệu báo cáo hoặc phiên hết hạn.", 404
    
    report_format = request.args.get('format', 'detail')
    path_to_download = report_file_paths.get(f"{report_format}_path")
    title_for_download = report_file_paths.get('display_title_for_filename', 'report')
    
    if not path_to_download or not os.path.exists(path_to_download):
        return f"Lỗi: File báo cáo '{report_format}' không tìm thấy.", 404

    clean_title_part = re.sub(r'[^\w\s-]', '', title_for_download.lower().replace(' ', '_'))[:50] or "report"
    download_file_name = f"{report_format}_{clean_title_part}.txt"
    try:
        return send_file(path_to_download, as_attachment=True, download_name=download_file_name, mimetype='text/plain; charset=utf-8')
    except Exception as e:
        app.logger.error(f"Lỗi gửi file báo cáo URL: {e}", exc_info=True); return "Lỗi gửi file.", 500

@app.route('/download_batch_excel/<excel_download_key>')
def download_batch_excel(excel_download_key):
    if not excel_download_key: return "Thiếu định danh file.", 400
    excel_file_path_to_send = session.get(excel_download_key)
    if not excel_file_path_to_send or not os.path.exists(excel_file_path_to_send):
        direct_path_try = os.path.join(TEMP_FILES_ROOT_DIR, excel_download_key)
        if os.path.exists(direct_path_try): excel_file_path_to_send = direct_path_try
        else: return "Không có dữ liệu file hoặc file đã bị xóa.", 404
    
    excel_basename = os.path.basename(excel_file_path_to_send)
    filename_prefix = re.sub(r'[^\w-]', '', excel_basename.replace("batch_results_", "").replace(".xlsx",""))[:20]
    final_excel_dl_name = f"ket_qua_phan_loai_{filename_prefix}.xlsx"
    try:
        return send_file(excel_file_path_to_send, as_attachment=True, download_name=final_excel_dl_name,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        app.logger.error(f"Lỗi gửi file Excel {excel_file_path_to_send}: {e}", exc_info=True)
        return "Lỗi gửi file.", 500


@app.route('/visualize_attention_phobert', methods=['POST'])
def visualize_attention_route_phobert():
    start_time = time.time()
    global tokenizer, model, device
    if not model or not tokenizer:
        return jsonify({"error": "Hệ thống chưa sẵn sàng (model/tokenizer)."}), 500

    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Dữ liệu không hợp lệ."}), 400
        text = data.get('text')
        layer_idx_str = data.get('layer', '-1'); head_idx_str = data.get('head', '0')
    except Exception as e:
        app.logger.error(f"Lỗi đọc JSON request: {e}")
        return jsonify({"error": "Lỗi xử lý dữ liệu đầu vào."}), 400

    if not text or not text.strip(): return jsonify({"error": "Vui lòng cung cấp văn bản."}), 400
    try:
        layer_idx = int(layer_idx_str); head_idx = int(head_idx_str)
    except ValueError: return jsonify({"error": "Layer và head phải là số."}), 400

    is_training_state = model.training
    if is_training_state: model.eval()
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MODEL_MAX_LEN, padding="max_length", return_attention_mask=True)
        input_ids_tensor = inputs['input_ids'].to(device)
        attention_mask_tensor = inputs['attention_mask'].to(device)
        
        token_ids_list = input_ids_tensor[0].tolist()
        token_strings_raw = tokenizer.convert_ids_to_tokens(token_ids_list)
        
        actual_len = attention_mask_tensor[0].sum().item()
        
        meaningful_tokens = []
        meaningful_indices_in_original_seq = []
        for i in range(actual_len):
            token_display = token_strings_raw[i].replace('@@', '').replace('Ġ', ' ').strip() 
            if token_strings_raw[i] == tokenizer.pad_token: continue
            if not token_display and token_strings_raw[i] == tokenizer.unk_token:
                 token_display = tokenizer.unk_token
            elif not token_display:
                token_display = token_strings_raw[i]

            meaningful_tokens.append(token_display if token_display else "[empty_token]")
            meaningful_indices_in_original_seq.append(i)

        if not meaningful_tokens: 
            return jsonify({"error": "Không có token nào để trực quan hóa."}), 400
        
        MAX_TOKENS_FOR_INTERACTIVE_PLOT = 100 
        if len(meaningful_tokens) > MAX_TOKENS_FOR_INTERACTIVE_PLOT:
            meaningful_tokens = meaningful_tokens[:MAX_TOKENS_FOR_INTERACTIVE_PLOT]
            meaningful_indices_in_original_seq = meaningful_indices_in_original_seq[:MAX_TOKENS_FOR_INTERACTIVE_PLOT]
            app.logger.info(f"Attention data truncated to {MAX_TOKENS_FOR_INTERACTIVE_PLOT} tokens for interactive display.")


        with torch.no_grad(): outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor)

        if not hasattr(outputs, 'attentions') or not outputs.attentions:
            return jsonify({"error": "Không tìm thấy attentions. Model cần output_attentions=True."}), 500

        num_model_layers = len(outputs.attentions)
        if layer_idx < 0: layer_idx = num_model_layers + layer_idx 
        if not (0 <= layer_idx < num_model_layers):
            return jsonify({"error": f"Layer không hợp lệ. Chọn từ {-num_model_layers} đến {num_model_layers-1}."}), 400
        
        selected_att_layer = outputs.attentions[layer_idx]
        num_model_heads = selected_att_layer.size(1)
        if not (0 <= head_idx < num_model_heads):
            return jsonify({"error": f"Head không hợp lệ cho layer {layer_idx}. Chọn từ 0 đến {num_model_heads-1}."}), 400

        att_scores_head = selected_att_layer[0, head_idx, :, :].cpu().numpy()
        
        meaningful_att_matrix = att_scores_head[np.ix_(meaningful_indices_in_original_seq, meaningful_indices_in_original_seq)]
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        app.logger.info(f"Thời gian xử lý cho visualize_attention_phobert: {processing_time_ms:.2f} ms")

        return jsonify({
            "tokens": meaningful_tokens,
            "attention_matrix": meaningful_att_matrix.tolist(),
            "processing_time_ms": processing_time_ms
        })

    except Exception as e:
        app.logger.error(f"Lỗi tạo dữ liệu attention: {e}", exc_info=True)
        return jsonify({"error": f"Không thể tạo dữ liệu attention: {str(e)}"}), 500
    finally:
        if is_training_state: model.train()

try:
    app.logger.info("Ứng dụng khởi tạo, tải tài nguyên...")
    load_resources()
    app.logger.info("Hoàn tất tải tài nguyên!")
except RuntimeError as e:
    print(f"Lỗi khởi tạo thất bại: {e}", file=sys.stderr); sys.exit(1)

if __name__ == '__main__':
    port_num = int(os.environ.get("PORT", 8080))
    app.logger.info(f"Chạy Flask server trên cổng {port_num}")
    app.run(host='0.0.0.0', port=port_num, debug=False)