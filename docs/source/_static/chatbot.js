document.addEventListener("DOMContentLoaded", () => {
  // ------------------------------------------------------------
  //  Utility: ensure 'marked' (markdown renderer) is available
  // ------------------------------------------------------------
  function ensureMarked() {
    return new Promise((resolve) => {
      if (window.marked) {
        resolve(window.marked);
        return;
      }

      // Load marked from CDN dynamically
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
      script.onload = () => resolve(window.marked);
      script.onerror = () => {
        console.warn("[chatbot] Failed to load marked.js; falling back to plain text.");
        resolve(null);
      };
      document.head.appendChild(script);
    });
  }

  // ------------------------------------------------------------
  //  DOM helpers
  // ------------------------------------------------------------
  function createElement(tag, className, text) {
    const el = document.createElement(tag);
    if (className) {
      el.className = className;
    }
    if (text !== undefined && text !== null) {
      el.textContent = text;
    }
    return el;
  }

  const launcher = createElement("button", "chatbot-launcher", "Chat");
  launcher.setAttribute("type", "button");

  const windowEl = createElement("div", "chatbot-window");
  const header = createElement("div", "chatbot-header");
  const title = createElement("span", "chatbot-title", "Docs Assistant");
  const closeBtn = createElement("button", "chatbot-close", "×");
  closeBtn.setAttribute("type", "button");

  header.appendChild(title);
  header.appendChild(closeBtn);

  const messagesEl = createElement("div", "chatbot-messages");

  const form = createElement("form", "chatbot-form");
  const input = createElement("input", "chatbot-input");
  input.setAttribute("type", "text");
  input.setAttribute("placeholder", "Ask documentation questions...");
  const sendBtn = createElement("button", "chatbot-send", "Send");
  sendBtn.setAttribute("type", "submit");

  form.appendChild(input);
  form.appendChild(sendBtn);

  windowEl.appendChild(header);
  windowEl.appendChild(messagesEl);
  windowEl.appendChild(form);

  document.body.appendChild(windowEl);
  document.body.appendChild(launcher);

  // ------------------------------------------------------------
  //  UI logic
  // ------------------------------------------------------------
  function toggleWindow(forceState) {
    const isOpen =
      forceState !== undefined ? forceState : !windowEl.classList.contains("open");
    if (isOpen) {
      windowEl.classList.add("open");
      input.focus();
    } else {
      windowEl.classList.remove("open");
    }
  }

  function addPlainMessage(text, role) {
    const message = createElement("div", `chatbot-message chatbot-${role}`);
    message.textContent = text;
    messagesEl.appendChild(message);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function renderMarkdown(markedLib, text) {
    if (!markedLib) {
      return null;
    }
    if (typeof markedLib.parse === "function") {
      return markedLib.parse(text);
    }
    if (typeof markedLib === "function") {
      return markedLib(text);
    }
    return null;
  }

  async function addMarkdownMessage(text, role) {
    const message = createElement("div", `chatbot-message chatbot-${role}`);

    const markedLib = await ensureMarked();
    const rendered = renderMarkdown(markedLib, text);
    if (rendered === null) {
      // Fallback: plain text
      message.textContent = text;
    } else {
      // Render markdown to HTML (works for new + legacy marked builds)
      message.innerHTML = rendered;
    }

    messagesEl.appendChild(message);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // sources: either list of strings or list of {source_id, corpus, source_path, url}
  function addSources(sources) {
    const container = createElement(
      "div",
      "chatbot-message chatbot-sources"
    );
    const label = createElement(
      "div",
      "chatbot-sources-label",
      "Sources:"
    );
    container.appendChild(label);

    sources.forEach((source) => {
      // Backwards compatibility: string sources
      if (typeof source === "string") {
        const line = createElement("div", "chatbot-source-line", source);
        container.appendChild(line);
        return;
      }

      // Structured source object
      const { source_id, corpus, source_path, url } = source;
      const line = createElement("div", "chatbot-source-line");
      const labelText =
        `${source_id ?? "?"} [${corpus ?? "?"}] — ${source_path ?? ""}`;

      if (url) {
        const link = document.createElement("a");
        link.textContent = labelText;
        link.href = url;
        link.target = "_blank";
        line.appendChild(link);
      } else {
        line.textContent = labelText;
      }

      container.appendChild(line);
    });

    messagesEl.appendChild(container);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setInputsDisabled(disabled) {
    input.disabled = disabled;
    sendBtn.disabled = disabled;
  }

  // ------------------------------------------------------------
  //  Networking
  // ------------------------------------------------------------
  async function sendMessage(event) {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) {
      return;
    }

    // User message: plain text
    addPlainMessage(message, "user");
    input.value = "";
    setInputsDisabled(true);

    try {
      // Same origin backend: /chat/
      const response = await fetch("/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();

      if (data && typeof data.answer === "string") {
        // BOT message: render as markdown
        await addMarkdownMessage(data.answer, "bot");
      } else {
        addPlainMessage("Unexpected response from server.", "bot");
      }

      if (Array.isArray(data?.sources) && data.sources.length > 0) {
        addSources(data.sources);
      }
    } catch (error) {
      const messageText =
        error instanceof Error ? error.message : "Unknown error";
      addPlainMessage(`Error: ${messageText}`, "bot");
    } finally {
      setInputsDisabled(false);
      input.focus();
    }
  }

  // ------------------------------------------------------------
  //  Event wiring
  // ------------------------------------------------------------
  launcher.addEventListener("click", () => toggleWindow());
  closeBtn.addEventListener("click", () => toggleWindow(false));
  form.addEventListener("submit", sendMessage);
});

// Include in conf.py:
// html_static_path = ["_static"]
// html_js_files = ["chatbot.js"]
// html_css_files = ["chatbot.css"]
