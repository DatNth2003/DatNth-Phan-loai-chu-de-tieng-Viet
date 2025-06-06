let classifyForm, submitButton, textInputContainerEl, urlInputContainerEl, fileInputContainerEl,
    textTitleInputEl, textInputEl, urlInputEl, fileInputEl, modeButtons,
    chunkStrategySelect, strategyDescriptionEl, activeInputModeField,
    attentionSourceTextDisplayEl, hiddenAttentionSourceTextEl,
    layerPhoBERTSelectEl, headPhoBERTSelectEl,
    attentionPhoBERTResultDiv, interactiveAttentionHeatmapDiv,
    attentionTimeInfoEl;

let currentMode;
let visualizeAttentionUrl;
let NO_CHUNK_TOKEN_LIMIT_JS = 256;
let CHUNK_SIZE_JS = 128;
let MODEL_MAX_LEN_JS = 256;
const INTERACTIVE_MAX_TOKEN_DISPLAY_JS = 100;

const strategyDescMap = {
    "no_chunking": `Không chia nhỏ văn bản. Phù hợp cho văn bản ngắn dưới ${NO_CHUNK_TOKEN_LIMIT_JS} token.`,
    "sum_logits": `Chia văn bản thành các đoạn ${CHUNK_SIZE_JS} token, tổng hợp điểm tin cậy (logits). Khuyến nghị cho văn bản dài.`,
    "mean_probabilities": "Chia văn bản, tính trung bình xác suất dự đoán từ các đoạn.",
    "max_probability": "Chọn dự đoán từ đoạn văn bản có xác suất chủ đề cao nhất.",
    "first_chunk": `Chỉ xem xét đoạn văn bản đầu tiên (${CHUNK_SIZE_JS} token) để dự đoán nhanh.`,
    "majority_vote": "Chia văn bản thành nhiều đoạn, mỗi đoạn dự đoán một chủ đề, chủ đề nào được đoán nhiều nhất sẽ thắng."
};

function updateStrategyDescription() {
    if (chunkStrategySelect && strategyDescriptionEl && classifyForm && classifyForm.dataset) {
        const selectedVal = chunkStrategySelect.value;
        let desc = strategyDescMap[selectedVal] || "Chọn một chiến lược.";
        desc = desc.replace('${NO_CHUNK_TOKEN_LIMIT_JS}', NO_CHUNK_TOKEN_LIMIT_JS.toString());
        desc = desc.replace('${CHUNK_SIZE_JS}', CHUNK_SIZE_JS.toString());
        strategyDescriptionEl.textContent = desc;
    }
}

function setInputMode(mode, clearValues = true) {
    currentMode = mode;
    if (activeInputModeField) activeInputModeField.value = mode;

    const localTextInputContainerEl = document.getElementById('textInputContainer');
    const localUrlInputContainerEl = document.getElementById('urlInputContainer');
    const localFileInputContainerEl = document.getElementById('fileInputContainer');
    const allContainers = [localTextInputContainerEl, localUrlInputContainerEl, localFileInputContainerEl];
    
    allContainers.forEach(c => { if (c) c.classList.remove('active'); });

    const localTextInputEl = document.getElementById('text_input_id');
    const localUrlInputEl = document.getElementById('url_input_id');
    const localFileInputEl = document.getElementById('file_input_id');
    const localTextTitleInputEl = document.getElementById('text_title_input_id');
    const allFields = [localTextInputEl, localUrlInputEl, localFileInputEl, localTextTitleInputEl];
    
    allFields.forEach(f => {
        if (f) {
            f.disabled = true;
            if (clearValues) f.value = (f.type === 'file') ? null : '';
        }
    });
    
    const currentModeButtons = document.querySelectorAll('.btn-mode-group .btn');
    if (currentModeButtons) {
        currentModeButtons.forEach(b => {
            if (b) b.classList.toggle('active', b.dataset.mode === mode);
        });
    }

    let activeContainer = null;
    let fieldsToEnable = [];

    if (mode === 'text') {
        activeContainer = localTextInputContainerEl;
        fieldsToEnable.push(localTextInputEl, localTextTitleInputEl);
    } else if (mode === 'url') {
        activeContainer = localUrlInputContainerEl;
        fieldsToEnable.push(localUrlInputEl);
    } else if (mode === 'file') {
        activeContainer = localFileInputContainerEl;
        fieldsToEnable.push(localFileInputEl);
    }

    if (activeContainer) activeContainer.classList.add('active');
    fieldsToEnable.forEach(f => { if (f) f.disabled = false; });
}

async function requestAndDisplayAttention(textToVisualize, layer, head) {
    const currentHiddenAttentionSourceTextEl = document.getElementById('hiddenAttentionSourceText');
    const currentLayerPhoBERTSelectEl = document.getElementById('layerPhoBERTSelect');
    const currentHeadPhoBERTSelectEl = document.getElementById('headPhoBERTSelect');
    const currentAttentionPhoBERTResultDiv = document.getElementById('attentionPhoBERTResult');
    const currentInteractiveAttentionHeatmapDiv = document.getElementById('interactiveAttentionHeatmap');
    const currentAttentionTimeInfoEl = document.getElementById('attentionTimeInfo');

    if (!currentHiddenAttentionSourceTextEl || !currentLayerPhoBERTSelectEl || !currentHeadPhoBERTSelectEl || !currentAttentionPhoBERTResultDiv || !currentInteractiveAttentionHeatmapDiv || !currentAttentionTimeInfoEl) {
        return;
    }
    if (!textToVisualize || !textToVisualize.trim() || currentLayerPhoBERTSelectEl.disabled) {
        currentAttentionPhoBERTResultDiv.style.display = 'none';
        if (typeof Plotly !== 'undefined' && currentInteractiveAttentionHeatmapDiv) Plotly.purge(currentInteractiveAttentionHeatmapDiv);
        currentAttentionTimeInfoEl.innerHTML = "";
        return;
    }

    const placeholderTextP = currentAttentionPhoBERTResultDiv.querySelector('p.text-muted.small:not(#attentionTimeInfo)');
    const originalPlaceholderText = 'Biểu đồ Heatmap Attention:';
    
    currentAttentionTimeInfoEl.innerHTML = `<span class="spinner-border spinner-border-sm text-info me-1" role="status" aria-hidden="true"></span>Đang xử lý...`;
    if(placeholderTextP) {
        placeholderTextP.innerHTML = `<i class="fas fa-spinner fa-spin me-1"></i>Đang tải dữ liệu attention...`;
    }
    if (typeof Plotly !== 'undefined' && currentInteractiveAttentionHeatmapDiv) Plotly.purge(currentInteractiveAttentionHeatmapDiv);
    currentAttentionPhoBERTResultDiv.style.display = 'block';

    const overallStartTime = performance.now();

    if (!visualizeAttentionUrl) {
        if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-exclamation-triangle text-danger"></i> Lỗi cấu hình URL client.`;
        currentAttentionTimeInfoEl.innerHTML = `<i class="fas fa-times-circle text-danger"></i> Lỗi`;
        return;
    }
    
    let networkLatency = 0;
    let ProcessingTime = 0;

    try {
        const fetchStartTime = performance.now();
        const response = await fetch(visualizeAttentionUrl, { 
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ text: textToVisualize, layer: layer, head: head })
        });
        const fetchEndTime = performance.now();
        networkLatency = fetchEndTime - fetchStartTime;

        if (response.ok) {
            const data = await response.json();
            if (data.processing_time_ms) {
                ProcessingTime = data.processing_time_ms;
            }

            if (data.error) {
                 if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-exclamation-triangle text-danger"></i> Lỗi từ server: ${data.error}`;
                 currentAttentionTimeInfoEl.innerHTML = `<i class="fas fa-times-circle text-danger"></i> Lỗi Server`;
                 return;
            }

            const tokens = data.tokens;
            const attentionMatrix = data.attention_matrix;

            if (!tokens || !attentionMatrix || tokens.length === 0 || attentionMatrix.length === 0) {
                if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-info-circle"></i> Không có đủ dữ liệu attention.`;
                currentAttentionTimeInfoEl.innerHTML = `<i class="fas fa-info-circle text-warning"></i> Không có dữ liệu`;
                return;
            }
            
            if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-spinner fa-spin me-1"></i>Đang vẽ biểu đồ...`;
            
            const reversedAttentionMatrix = attentionMatrix.slice().reverse();
            const plotlyData = [{
                x: tokens,
                y: tokens, 
                z: reversedAttentionMatrix, 
                type: 'heatmap',
                colorscale: 'Viridis',
                showscale: true,
                hoverongaps: false,
                hovertemplate: '<b>Truy vấn (Y)</b>: %{y}<br>' +
                               '<b>Khóa (X)</b>: %{x}<br>' +
                               '<b>Attention</b>: %{z:.4f}<extra></extra>'
            }];

            const layout = {
                title: { text: `Heatmap Attention (Layer: ${layer}, Head: ${head})`, font: { size: 16 } },
                xaxis: { side: 'top', tickangle: 0, automargin: true, tickfont: {size: Math.max(6, 10 - Math.floor(tokens.length / 15))}, autorange: true },
                yaxis: { autorange: 'reversed', automargin: true, tickfont: {size: Math.max(6, 10 - Math.floor(tokens.length / 15))} },
                height: Math.max(450, tokens.length * 15 + 200),
                margin: { t: 80, l: 100, b: 50, r: 50 },
                paper_bgcolor: '#f9f9f9',
                plot_bgcolor: '#f9f9f9'
            };
            
            const plotStartTime = performance.now();
            setTimeout(() => {
                if (typeof Plotly !== 'undefined' && currentInteractiveAttentionHeatmapDiv) {
                    Plotly.newPlot(currentInteractiveAttentionHeatmapDiv, plotlyData, layout, {responsive: true});
                    const plotEndTime = performance.now();
                    const plotRenderTime = plotEndTime - plotStartTime;
                    if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-chart-area"></i> ${originalPlaceholderText}`;
                    
                    const overallEndTime = performance.now();
                    const totalUserWaitTime = overallEndTime - overallStartTime;

                    if (currentAttentionTimeInfoEl) {
                        currentAttentionTimeInfoEl.innerHTML = `
                            <i class="fas fa-network-wired text-primary"></i> Mạng: ${(networkLatency / 1000).toFixed(2)}s | 
                            <i class="fas fa-server text-success"></i> Server: ${(ProcessingTime / 1000).toFixed(2)}s |
                            <i class="fas fa-palette text-info"></i> Vẽ: ${(plotRenderTime / 1000).toFixed(2)}s |
                            <i class="far fa-clock text-secondary"></i> Tổng chờ: ${(totalUserWaitTime / 1000).toFixed(2)}s
                        `;
                    }

                } else {
                     if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-exclamation-triangle text-danger"></i> Lỗi tải thư viện biểu đồ.`;
                     if(currentAttentionTimeInfoEl) currentAttentionTimeInfoEl.innerHTML = `<i class="fas fa-times-circle text-danger"></i> Lỗi vẽ`;
                }
            }, 50);

        } else {
            const errorText = await response.text(); 
            if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-exclamation-triangle text-danger"></i> Lỗi tải dữ liệu: ${response.status}`;
            if(currentAttentionTimeInfoEl) currentAttentionTimeInfoEl.innerHTML = `<i class="fas fa-times-circle text-danger"></i> Lỗi tải`;
        }
    } catch (error) {
        if(placeholderTextP) placeholderTextP.innerHTML = `<i class="fas fa-exclamation-triangle text-danger"></i> Lỗi kết nối.`;
        const overallEndTimeCatch = performance.now();
        const totalUserWaitTimeCatch = overallEndTimeCatch - overallStartTime;
        if(currentAttentionTimeInfoEl) currentAttentionTimeInfoEl.innerHTML = `<i class="fas fa-times-circle text-danger"></i> Lỗi (${(totalUserWaitTimeCatch/1000).toFixed(2)}s)`;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    classifyForm = document.getElementById('classifyForm');
    submitButton = document.getElementById('submitButton');
    textInputContainerEl = document.getElementById('textInputContainer');
    urlInputContainerEl = document.getElementById('urlInputContainer');
    fileInputContainerEl = document.getElementById('fileInputContainer');
    textTitleInputEl = document.getElementById('text_title_input_id');
    textInputEl = document.getElementById('text_input_id');
    urlInputEl = document.getElementById('url_input_id');
    fileInputEl = document.getElementById('file_input_id');
    modeButtons = document.querySelectorAll('.btn-mode-group .btn');
    chunkStrategySelect = document.getElementById('chunk_strategy_select');
    strategyDescriptionEl = document.getElementById('strategyDescription');
    activeInputModeField = document.getElementById('active_input_mode_field');
    
    attentionSourceTextDisplayEl = document.getElementById('attentionSourceTextDisplay');
    hiddenAttentionSourceTextEl = document.getElementById('hiddenAttentionSourceText');
    layerPhoBERTSelectEl = document.getElementById('layerPhoBERTSelect');
    headPhoBERTSelectEl = document.getElementById('headPhoBERTSelect');
    attentionPhoBERTResultDiv = document.getElementById('attentionPhoBERTResult');
    interactiveAttentionHeatmapDiv = document.getElementById('interactiveAttentionHeatmap');
    attentionTimeInfoEl = document.getElementById('attentionTimeInfo');
    classificationTimeInfoEl = document.getElementById('classificationTimeInfo');


    if (classifyForm && classifyForm.dataset) { 
        visualizeAttentionUrl = classifyForm.dataset.visualizeUrl;
        NO_CHUNK_TOKEN_LIMIT_JS = parseInt(classifyForm.dataset.noChunkTokenLimit) || 256;
        CHUNK_SIZE_JS = parseInt(classifyForm.dataset.chunkSize) || 128;
        MODEL_MAX_LEN_JS = parseInt(classifyForm.dataset.modelMaxLen) || 256;
        
        const modelMaxLenDisplay = document.getElementById('modelMaxLengthTokenDisplay');
        if(modelMaxLenDisplay) modelMaxLenDisplay.textContent = MODEL_MAX_LEN_JS;
        
        const interactiveMaxTokenDisplayEl = document.getElementById('interactiveMaxTokenDisplay');
        if(interactiveMaxTokenDisplayEl) { 
             interactiveMaxTokenDisplayEl.textContent = INTERACTIVE_MAX_TOKEN_DISPLAY_JS;
        }

        if(chunkStrategySelect){
            chunkStrategySelect.value = classifyForm.dataset.defaultStrategy || "sum_logits";
            chunkStrategySelect.addEventListener('change', updateStrategyDescription);
        }
        updateStrategyDescription(); 
    } else {
        console.error("Lỗi: Không tìm thấy #classifyForm hoặc dataset của nó. Kiểm tra lại ID và data-attributes trong HTML.");
    }
    
    currentMode = activeInputModeField ? activeInputModeField.value : 'url';
    setInputMode(currentMode, false); 
    
    if(modeButtons) {
        modeButtons.forEach(button => {
            button.addEventListener('click', function() { 
                setInputMode(this.dataset.mode, true); 
            });
        });
    }
    
    const resultsArea = document.getElementById('resultsArea');
    const attentionCardArea = document.getElementById('attentionCard');
    if (resultsArea && (resultsArea.innerHTML || '').trim().length > 0 && window.location.hash === '#resultsArea') {
         setTimeout(() => { 
            const targetElement = resultsArea.offsetParent === null && attentionCardArea ? attentionCardArea : resultsArea; 
            if (targetElement) targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' }); 
        }, 150);
    }

    document.querySelectorAll('a[data-bs-toggle="collapse"]').forEach(ct => {
        const icon = ct.querySelector('.fa-chevron-down, .fa-chevron-up');
        if(icon){
            const targetCollapse = document.querySelector(ct.getAttribute('href'));
            if(targetCollapse){
                const updateIcon = () => {
                    if (targetCollapse.classList.contains('show')) {
                        icon.classList.remove('fa-chevron-down');
                        icon.classList.add('fa-chevron-up');
                         ct.setAttribute('aria-expanded', 'true');
                    } else {
                        icon.classList.remove('fa-chevron-up');
                        icon.classList.add('fa-chevron-down');
                        ct.setAttribute('aria-expanded', 'false');
                    }
                };
                targetCollapse.addEventListener('show.bs.collapse', updateIcon);
                targetCollapse.addEventListener('hide.bs.collapse', updateIcon);
                if (targetCollapse.classList.contains('show')) {
                    icon.classList.remove('fa-chevron-down');
                    icon.classList.add('fa-chevron-up');
                    ct.setAttribute('aria-expanded', 'true');
                } else {
                     icon.classList.remove('fa-chevron-up');
                     icon.classList.add('fa-chevron-down');
                     ct.setAttribute('aria-expanded', 'false');
                }
            }
        }
    });
    
    const initialTextForAttention = hiddenAttentionSourceTextEl ? hiddenAttentionSourceTextEl.value : ""; 

    function setupAttentionInterface(textForViz) {
        if (!attentionSourceTextDisplayEl || !layerPhoBERTSelectEl || !headPhoBERTSelectEl || !attentionPhoBERTResultDiv || !interactiveAttentionHeatmapDiv ) {
            return; 
        }

        if (textForViz && textForViz.trim() !== "") {
            const displayText = textForViz.length > 150 ? textForViz.substring(0, 150) + "..." : textForViz;
            attentionSourceTextDisplayEl.textContent = displayText;
            
            layerPhoBERTSelectEl.disabled = false;
            headPhoBERTSelectEl.disabled = false;

            const defaultLayer = layerPhoBERTSelectEl.value;
            const defaultHead = headPhoBERTSelectEl.value;
            requestAndDisplayAttention(textForViz, defaultLayer, defaultHead);
        } else {
            attentionSourceTextDisplayEl.innerHTML = "<i>Chưa có văn bản nào được phân loại.</i>";
            attentionPhoBERTResultDiv.style.display = 'none';
            if (typeof Plotly !== 'undefined' && interactiveAttentionHeatmapDiv) Plotly.purge(interactiveAttentionHeatmapDiv);
            layerPhoBERTSelectEl.disabled = true;
            headPhoBERTSelectEl.disabled = true;
        }
    }

    setupAttentionInterface(initialTextForAttention);

    if (layerPhoBERTSelectEl) {
        layerPhoBERTSelectEl.addEventListener('change', function() {
            const currentText = hiddenAttentionSourceTextEl ? hiddenAttentionSourceTextEl.value : "";
            requestAndDisplayAttention(currentText, this.value, headPhoBERTSelectEl.value);
        });
    }
    if (headPhoBERTSelectEl) {
        headPhoBERTSelectEl.addEventListener('change', function() {
            const currentText = hiddenAttentionSourceTextEl ? hiddenAttentionSourceTextEl.value : "";
            requestAndDisplayAttention(currentText, layerPhoBERTSelectEl.value, this.value);
        });
    }
});


if (classifyForm && submitButton) {
    classifyForm.addEventListener('submit', function(event) {
        let hasContent = false;
        if (currentMode === 'text' && ((textInputEl && textInputEl.value.trim()) || (textTitleInputEl && textTitleInputEl.value.trim()))) hasContent = true;
        else if (currentMode === 'url' && urlInputEl && urlInputEl.value.trim().match(/^(https?:\/\/[^\s]+)$/)) hasContent = true;
        else if (currentMode === 'file' && fileInputEl && fileInputEl.files.length > 0) hasContent = true;

        if (!hasContent) {
            alert(currentMode === 'url' && urlInputEl && urlInputEl.value.trim() && !urlInputEl.value.match(/^(https?:\/\/[^\s]+)$/) ? 
                  "URL không hợp lệ (cần http:// hoặc https://)." : 
                  "Vui lòng cung cấp nội dung (URL, văn bản, hoặc file).");
            event.preventDefault(); return;
        }
        if (textInputEl) textInputEl.name = (currentMode === 'text') ? 'text' : '';
        if (textTitleInputEl) textTitleInputEl.name = (currentMode === 'text') ? 'text_title' : '';
        if (urlInputEl) urlInputEl.name = (currentMode === 'url') ? 'url' : '';
        if (fileInputEl) fileInputEl.name = (currentMode === 'file') ? 'file_input' : '';
        
        sessionStorage.setItem('classificationSubmitTime', performance.now().toString());
        submitButton.disabled = true;
        submitButton.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang phân loại...`;
    });
}

const suggestionItems = document.querySelectorAll('.suggestion-item-simple');
if(suggestionItems) {
    suggestionItems.forEach(item => {
        item.addEventListener('click', function(event) {
            event.preventDefault(); 
            const sugUrl = this.dataset.url, sugTitle = this.dataset.title, sugDesc = this.dataset.description || ""; 
            if (sugUrl && sugUrl !== '#') {
                setInputMode('url', true);
                if(urlInputEl) { urlInputEl.value = sugUrl; urlInputEl.focus(); }
            } else {
                setInputMode('text', true); 
                if(textTitleInputEl) textTitleInputEl.value = sugTitle; 
                if(textInputEl) { textInputEl.value = sugDesc.trim(); textInputEl.focus(); }
            }
            const formCol = document.getElementById('form-column');
            if(formCol) formCol.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
}

window.addEventListener('load', function() {
    const submitTimeStr = sessionStorage.getItem('classificationSubmitTime');
    if (submitTimeStr && classificationTimeInfoEl) {
        const submitTime = parseFloat(submitTimeStr);
        const loadTime = performance.now();
        const pageLoadLatency = loadTime - submitTime;
                
        const TimeMsText = classificationTimeInfoEl.textContent;
        let TimeMs = 0;
        if (TimeMsText && TimeMsText.includes("Server:")) {
            const match = TimeMsText.match(/(\d+\.\d+)/);
            if (match && match[1]) {
                TimeMs = parseFloat(match[1]) * 1000;
            }
        }

        let clientSideProcessingTime = pageLoadLatency - TimeMs;
        if (clientSideProcessingTime < 0) clientSideProcessingTime = 0;

        let timeInfoHTML = "";
        if (TimeMs > 0) {
             timeInfoHTML += `<i class="fas fa-server text-success"></i> Server: ${(TimeMs / 1000).toFixed(2)}s`;
        }
        timeInfoHTML += `${TimeMs > 0 ? ' | ' : ''}<i class="far fa-hourglass text-info"></i> Tổng: ${(pageLoadLatency / 1000).toFixed(2)}s`;
        
        classificationTimeInfoEl.innerHTML = timeInfoHTML;
        sessionStorage.removeItem('classificationSubmitTime');
    }
});
