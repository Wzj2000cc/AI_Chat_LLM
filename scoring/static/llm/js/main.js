$(document).ready(function () {
    marked.setOptions({
        renderer: new marked.Renderer(),
        highlight: function (code, lang) {
            const hljs = require('highlight.js');
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, {language}).value;
        },
        langPrefix: 'hljs language-', // highlight.js css expects a top-level 'hljs' class.
        pedantic: false,
        gfm: true,
        breaks: false,
        sanitize: false,
        smartypants: false,
        xhtml: false
    });

    async function postData(url, data) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: data
            });

            // 创建一个新的容器
            const answerDiv = $('<div class="message answer"></div>');
            $('#chat-box').append(answerDiv);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let buffer = '';
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, {stream: true});

                const events = buffer.split('\n\n');
                buffer = events.pop();

                for (const event of events) {
                    if (event.startsWith('data: ')) {
                        try {
                            const jsonResponse = JSON.parse(event.slice(5));
                            const message = jsonResponse.message;

                            // 使用 marked 解析 Markdown
                            const htmlContent = marked.parse(message);

                            // 使用 jQuery 的 html 方法来更新 HTML 内容
                            answerDiv.html(htmlContent);

                            $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
                        } catch (e) {
                            console.error('Error parsing JSON:', e);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Error:', error);
            // 可以在这里显示用户友好的错误消息
        }
    }

    function sendMessage(url) {
        // 获取表单数据
        const Scene = $('#scene').val();
        const esKnowledge = $('#embedding_es').val();
        const milvusKnowledge = $('#embedding_milvus').val();
        const llmModel = $('#llm_model').val();
        const coarseRecall = $('#cu_count').val() || 10; // 默认值10
        const fineRecall = $('#jg_count').val() || 5; // 默认值5
        const rewriteCount = $('#rewrite_count').val() || 0; // 默认值0
        const Temperature = $('#temperature').val() || 0.01; // 默认值0.01
        const message = $('#message-input').val().trim();

        if (message !== '') {
            const requestData = {
                scene: Scene,
                esKnowledge: esKnowledge,
                milvusKnowledge: milvusKnowledge,
                llmModel: llmModel,
                coarseRecall: coarseRecall,
                fineRecall: fineRecall,
                rewriteCount: rewriteCount,
                message: message,
                temperature: Temperature
            };

            // 显示发送的消息在右侧
            $('#chat-box').append('<div class="message question">' + message.split('\n').join('<br/>\n') + '</div>');
            $('#message-input').val('');
            $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);

            // 发送请求
            postData(url, JSON.stringify(requestData));
        }
    }

    // function sendMessage(url) {
    //     const message = $('#message-input').val();
    //     const id = $('#hiddenID').text();
    //     if (message.trim() !== '') {
    //         htmlmessage = message.split('\n').join('<br/>\n');
    //         // 显示发送的消息在右侧
    //         $('#chat-box').append('<div class="message question">' + htmlmessage + '</div>');
    //         $('#message-input').val('');
    //         $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
    //         postData(url, JSON.stringify({query: message.trim(), id: id}));
    //     }
    // }

    // function postData(url, data) {
    //     // 发送消息到后端
    //     fetch(url, {
    //         method: 'POST',
    //         headers: {
    //             'Content-Type': 'application/json'
    //         },
    //         body: data
    //     }).then(response => {
    //         const reader = response.body.getReader();
    //         const decoder = new TextDecoder();
    //         let accumulatedMessage = '';
    //         let answerDiv = $('<div class="message answer"></div>'); // 创建一个容器用于累积文本
    //         $('#chat-box').append(answerDiv);
    //
    //         function processStream() {
    //             reader.read().then(({done, value}) => {
    //                 if (done) {
    //                     return;
    //                 }
    //                 accumulatedMessage += decoder.decode(value, {stream: true});
    //
    //                 // 将累积的文本逐字追加到容器中
    //                 // 使用 marked 解析 Markdown
    //                 const htmlContent = marked.parse(accumulatedMessage);
    //                 answerDiv.html(htmlContent);
    //
    //                 $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
    //
    //                 processStream();
    //             });
    //         }
    //
    //         processStream();
    //     }).catch(error => {
    //         console.error('Error:', error);
    //     });
    // }


    // 点击发送按钮时发送消息
    $('#send-button').click(function () {
        sendMessage('/scoring/llm_chat/');
    });


    // 按下 Ctrl + Enter 发送消息
    $('#message-input').keydown(function (event) {
        if (event.ctrlKey && event.keyCode == 13) {
            sendMessage('/scoring/llm_chat/');
        }
    });


    const sendButton = $('#send-button');
    const sideButtons = $('#side-buttons');

    // 显示侧边按钮
    function showSideButtons() {
        sideButtons.css({
            opacity: 1,
            visibility: 'visible'
        });
    }

    // 隐藏侧边按钮
    function hideSideButtons() {
        sideButtons.css({
            opacity: 0,
            visibility: 'hidden'
        });
    }

    // 处理发送按钮的悬停
    sendButton.on('mouseenter', function () {
        showSideButtons();
    });

    // 处理侧边按钮的悬停
    sideButtons.on('mouseenter', function () {
        showSideButtons();
    });

    // 处理鼠标离开发送按钮和侧边按钮
    sendButton.add(sideButtons).on('mouseleave', function () {
        setTimeout(() => {
            if (!sideButtons.is(':hover') && !sendButton.is(':hover')) {
                hideSideButtons();
            }
        }, 100); // 设置一个短暂的延迟以确保用户能够移动鼠标到侧边按钮上
    });
});

function tj_serach(qus) {
    $('#message-input').val(qus);
    $('#send-button').click();
}

function tj_chart() {
    jQuery.ajax({
        type: "POST",
        url: "/tj_chart",
        dataType: "text",
        success: function (data) {
            console.log(data);
        }
    });
}