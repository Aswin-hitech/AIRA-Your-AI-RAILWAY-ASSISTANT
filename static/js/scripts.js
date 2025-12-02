const messagesContainer = document.getElementById('messages');
const sendBtn = document.getElementById('send-btn');
const inputMessage = document.getElementById('input-message');
const attachBtn = document.getElementById('attach-btn');
const voiceBtn = document.getElementById('voice-btn');
let isSending = false;

/* ----------------------------------------
    MESSAGE & UI HANDLING
----------------------------------------- */

// Add message bubble
function addMessage(content, sender='bot-msg') {
  const div = document.createElement('div');
  div.className = `message ${sender}`;

  const lines = content.split('<br>');
  lines.forEach(line => {
    const p = document.createElement('p');
    p.innerHTML = line;
    div.appendChild(p);
  });

  messagesContainer.appendChild(div);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Add quick reply buttons
function addQuickReplies(options) {
  const container = document.createElement('div');
  container.className = 'quick-replies';

  options.forEach(opt => {
    const btn = document.createElement('button');
    btn.className = 'quick-btn';
    btn.textContent = opt;

    btn.onclick = () => {
      addMessage(opt, 'user-msg');
      sendMessage(opt);
      container.remove();
    };

    container.appendChild(btn);
  });

  messagesContainer.appendChild(container);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Typing animation
function showTyping() {
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.innerHTML = '<span></span><span></span><span></span>';
  messagesContainer.appendChild(typing);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return typing;
}

/* ----------------------------------------
    SEND MESSAGE TO BACKEND
----------------------------------------- */

function sendMessage(msg) {
  if (isSending) return;
  isSending = true;
  const typing = showTyping();

  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: msg })
  })
  .then(r => r.json())
  .then(data => {
    typing.remove();

    if (data.reply) addMessage(data.reply, 'bot-msg');
    if (data.buttons && data.buttons.length > 0) addQuickReplies(data.buttons);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  })
  .catch(() => {
    typing.remove();
    addMessage('Error occurred. Try again.', 'bot-msg');
  })
  .finally(() => { isSending = false; });
}

/* ----------------------------------------
    VOICE RECORDING WITH POPUP UI
----------------------------------------- */

let recorder;
let chunks = [];
let isRecording = false;

const popup = document.getElementById("voice-popup");
const voiceStatus = document.getElementById("voice-status");
const stopRecordBtn = document.getElementById("stop-record-btn");

async function startRecording() {
    try {
        let stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        chunks = [];

        // Show popup
        popup.style.display = "block";
        voiceStatus.innerText = "🔴 Listening… Speak now";

        recorder.ondataavailable = e => {
            if (e.data.size > 0) chunks.push(e.data);
        };

recorder.onstop = async () => {
    popup.style.display = "none";  // close popup

    let blob = new Blob(chunks, { type: "audio/webm" });
    let reader = new FileReader();

    reader.onloadend = async () => {
        let base64Audio = reader.result.split(",")[1];

        let response = await fetch("/voice_to_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio: base64Audio })
        });

        let data = await response.json();

        if (data.success) {
            inputMessage.value = data.text;    // Insert text
            sendBtn.click();                   // 🚀 AUTO-SEND
        } else {
            addMessage("🎤 Error: " + data.error, "bot-msg");
        }
    };

    reader.readAsDataURL(blob);
};


        recorder.start();
        isRecording = true;

    } catch (err) {
        popup.style.display = "none";
        addMessage("❌ Microphone permission denied.", "bot-msg");
    }
}

function stopRecording() {
    if (recorder && recorder.state !== "inactive") {
        recorder.stop();
    }
    isRecording = false;
}

// Button inside popup
stopRecordBtn.onclick = () => {
    stopRecording();
};

// Mic icon in chat
voiceBtn.onclick = () => {
    if (!isRecording) startRecording();
    else stopRecording();
};



/* ----------------------------------------
    SEND BUTTON & ENTER KEY
----------------------------------------- */

sendBtn.addEventListener('click', () => {
  const msg = inputMessage.value.trim();
  if (!msg || isSending) return;

  addMessage(msg, 'user-msg');
  inputMessage.value = '';
  sendMessage(msg);
});

inputMessage.addEventListener('keypress', e => {
  if (e.key === 'Enter') {
    sendBtn.click();
    e.preventDefault();
  }
});

/* ----------------------------------------
    ATTACH BUTTON (still placeholder)
----------------------------------------- */

attachBtn.addEventListener('click', () => {
  alert('Attachments coming soon!');
});

/* ----------------------------------------
    LOAD INITIAL MESSAGE
----------------------------------------- */

window.addEventListener('load', () => {
  const typing = showTyping();

  setTimeout(() => {
    fetch('/chat_default')
      .then(r => r.json())
      .then(data => {
        typing.remove();
        if (data.reply) addMessage(data.reply, 'bot-msg');
        if (data.buttons) addQuickReplies(data.buttons);
      })
      .catch(() => {
        typing.remove();
        addMessage('Welcome to AIRA! How can I assist you?', 'bot-msg');
      });
  }, 800);
});
