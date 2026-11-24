(() => {
  "use strict";

  const CONFIG_URL = "/model_organizer/v1/config";
  const META_URL = "/model_organizer/v1/card_meta";
  const CONFIG_TTL_MS = 15000;

  let configState = {
    enable_card_buttons: true,
    blur_nsfw_gallery: true,
    domain: "civitai.com",
    version: "1.1.0",
  };
  let lastConfigFetch = 0;
  let configPromise = null;
  const metaCache = new Map();

  let toastTimer = null;
  let activeModal = null;

  function rootApp() {
    if (typeof window.gradioApp === "function") {
      try {
        return window.gradioApp();
      } catch (_err) {
        return document;
      }
    }
    return document;
  }

  async function refreshConfig(force = false) {
    const now = Date.now();
    if (!force && lastConfigFetch > 0 && now - lastConfigFetch < CONFIG_TTL_MS) {
      return configState;
    }
    if (configPromise) {
      return configPromise;
    }
    configPromise = fetch(CONFIG_URL, { cache: "no-store" })
      .then((resp) => (resp.ok ? resp.json() : null))
      .then((data) => {
        if (data && typeof data === "object") {
          configState = {
            enable_card_buttons: data.enable_card_buttons !== false,
            blur_nsfw_gallery: data.blur_nsfw_gallery !== false,
            domain: data.domain || "civitai.com",
            version: data.version || "1.1.0",
          };
          lastConfigFetch = Date.now();
          if (!configState.enable_card_buttons) {
            document.querySelectorAll(".mo-card-toolbar").forEach((el) => {
              el.style.display = "none";
            });
          } else {
            document.querySelectorAll(".mo-card-toolbar").forEach((el) => {
              el.style.display = "";
            });
          }
        }
        return configState;
      })
      .catch(() => configState)
      .finally(() => {
        configPromise = null;
      });
    return configPromise;
  }

  function showToast(message, isError = false) {
    let el = document.getElementById("mo-toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "mo-toast";
      el.className = "mo-toast";
      document.body.appendChild(el);
    }
    el.textContent = message;
    el.classList.toggle("mo-toast-error", Boolean(isError));
    el.classList.add("mo-toast-show");
    if (toastTimer) {
      clearTimeout(toastTimer);
    }
    toastTimer = setTimeout(() => {
      el.classList.remove("mo-toast-show");
    }, 2600);
  }

  function parseCardContext(card) {
    if (!card) return null;
    const container = card.closest('[id$="_cards"]');
    if (!container || !container.id) return null;
    const cid = container.id.toLowerCase();

    let tabKind = "";
    if (cid.startsWith("txt2img_")) {
      tabKind = "txt2img";
    } else if (cid.startsWith("img2img_")) {
      tabKind = "img2img";
    } else {
      return null;
    }

    let modelType = "";
    if (cid.includes("lora")) {
      modelType = "lora";
    } else if (cid.includes("checkpoint")) {
      modelType = "checkpoint";
    } else if (cid.includes("textual_inversion")) {
      modelType = "embedding";
    } else if (cid.includes("lycoris")) {
      modelType = "lycoris";
    } else {
      return null;
    }

    const name = (card.getAttribute("data-name") || card.querySelector(".name")?.textContent || "").trim();
    const copyBtn = card.querySelector(".copy-path-button");
    const pathHint = (
      card.getAttribute("data-sort-path") ||
      (copyBtn ? copyBtn.getAttribute("data-clipboard-text") : "") ||
      card.getAttribute("data-path") ||
      ""
    ).trim();

    if (!name && !pathHint) return null;
    return { tabKind, modelType, name, pathHint };
  }

  async function fetchCardMeta(ctx) {
    const cacheKey = `${ctx.modelType}|${ctx.pathHint}|${ctx.name}|${configState.domain}`;
    if (metaCache.has(cacheKey)) {
      return metaCache.get(cacheKey);
    }
    const qs = new URLSearchParams({
      model_type: ctx.modelType,
      name: ctx.name,
      path: ctx.pathHint,
    });
    const resp = await fetch(`${META_URL}?${qs.toString()}`, { cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    metaCache.set(cacheKey, data);
    return data;
  }

  function setTextareaValue(textarea, value) {
    if (!textarea) return;
    const proto = window.HTMLTextAreaElement && window.HTMLTextAreaElement.prototype;
    const desc = proto ? Object.getOwnPropertyDescriptor(proto, "value") : null;
    if (desc && typeof desc.set === "function") {
      desc.set.call(textarea, value);
    } else {
      textarea.value = value;
    }
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    textarea.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function getPromptTextarea(tabKind, negative = false) {
    const root = rootApp();
    const sel = negative ? `#${tabKind}_neg_prompt textarea` : `#${tabKind}_prompt textarea`;
    return root.querySelector(sel) || document.querySelector(sel);
  }

  function splitPromptTokens(text) {
    return String(text || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function appendPromptTokens(tabKind, addition, negative = false) {
    const cleanAdd = String(addition || "").trim();
    if (!cleanAdd) return 0;
    const ta = getPromptTextarea(tabKind, negative);
    if (!ta) {
      throw new Error(`未找到 ${tabKind} 提示词输入框`);
    }
    const existing = ta.value || "";
    const existingTokens = new Set(splitPromptTokens(existing).map((t) => t.toLowerCase()));
    const incoming = splitPromptTokens(cleanAdd);
    const toAdd = [];
    for (const tok of incoming) {
      const low = tok.toLowerCase();
      if (!existingTokens.has(low)) {
        existingTokens.add(low);
        toAdd.push(tok);
      }
    }
    if (toAdd.length === 0) {
      return 0;
    }
    const trimmed = existing.trim();
    const joinStr = toAdd.join(", ");
    let nextVal = joinStr;
    if (trimmed) {
      const sep = trimmed.endsWith(",") ? " " : ", ";
      nextVal = `${trimmed}${sep}${joinStr}`;
    }
    setTextareaValue(ta, nextVal);
    return toAdd.length;
  }

  function replacePrompts(tabKind, positive, negative) {
    const posTa = getPromptTextarea(tabKind, false);
    if (!posTa) {
      throw new Error(`未找到 ${tabKind} 正向提示词框`);
    }
    setTextareaValue(posTa, String(positive || ""));
    if (typeof negative === "string") {
      const negTa = getPromptTextarea(tabKind, true);
      if (negTa) {
        setTextareaValue(negTa, negative);
      }
    }
  }

  function isSafeHttpUrl(url) {
    return typeof url === "string" && /^https?:\/\//i.test(url.trim());
  }

  async function copyText(text) {
    if (!text) return false;
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (_err) {
        // fallback below
      }
    }
    const tmp = document.createElement("textarea");
    tmp.value = text;
    tmp.style.position = "fixed";
    tmp.style.left = "-9999px";
    document.body.appendChild(tmp);
    tmp.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (_err) {
      ok = false;
    }
    document.body.removeChild(tmp);
    return ok;
  }

  function closeModal() {
    if (activeModal && activeModal.parentNode) {
      activeModal.parentNode.removeChild(activeModal);
    }
    activeModal = null;
  }

  function openGalleryModal(meta, ctx) {
    closeModal();

    const overlay = document.createElement("div");
    overlay.className = "mo-gallery-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");

    const dialog = document.createElement("div");
    dialog.className = "mo-gallery-dialog";

    // Header
    const header = document.createElement("div");
    header.className = "mo-gallery-header";

    const titleWrap = document.createElement("div");
    titleWrap.className = "mo-gallery-title-wrap";

    const titleEl = document.createElement("h3");
    titleEl.className = "mo-gallery-title";
    titleEl.textContent = meta.model_name || meta.stem || ctx.name;
    titleWrap.appendChild(titleEl);

    const subBadges = document.createElement("div");
    subBadges.className = "mo-gallery-subbadges";
    if (meta.version_name) {
      const b = document.createElement("span");
      b.className = "mo-badge";
      b.textContent = meta.version_name;
      subBadges.appendChild(b);
    }
    if (meta.base_model) {
      const b = document.createElement("span");
      b.className = "mo-badge mo-badge-accent";
      b.textContent = meta.base_model;
      subBadges.appendChild(b);
    }
    titleWrap.appendChild(subBadges);
    header.appendChild(titleWrap);

    const hdrActions = document.createElement("div");
    hdrActions.className = "mo-gallery-header-actions";
    if (isSafeHttpUrl(meta.civitai_url)) {
      const extLink = document.createElement("a");
      extLink.className = "mo-btn mo-btn-secondary";
      extLink.href = meta.civitai_url;
      extLink.target = "_blank";
      extLink.rel = "noopener noreferrer";
      extLink.textContent = "🌐 打开 C 站原页 ↗";
      hdrActions.appendChild(extLink);
    }
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "mo-gallery-close";
    closeBtn.textContent = "×";
    closeBtn.title = "关闭 (ESC)";
    closeBtn.addEventListener("click", closeModal);
    hdrActions.appendChild(closeBtn);
    header.appendChild(hdrActions);
    dialog.appendChild(header);

    // Body
    const body = document.createElement("div");
    body.className = "mo-gallery-body";

    const images = Array.isArray(meta.images) ? meta.images : [];
    let currentIdx = 0;
    const revealed = new Set();

    // Left: media viewer
    const mediaCol = document.createElement("div");
    mediaCol.className = "mo-gallery-media";

    const stage = document.createElement("div");
    stage.className = "mo-gallery-stage";
    mediaCol.appendChild(stage);

    const navBar = document.createElement("div");
    navBar.className = "mo-gallery-nav";
    const prevBtn = document.createElement("button");
    prevBtn.type = "button";
    prevBtn.className = "mo-btn mo-btn-secondary";
    prevBtn.textContent = "◀ 上一张";
    const counterEl = document.createElement("span");
    counterEl.className = "mo-gallery-counter";
    const nextBtn = document.createElement("button");
    nextBtn.type = "button";
    nextBtn.className = "mo-btn mo-btn-secondary";
    nextBtn.textContent = "下一张 ▶";
    navBar.append(prevBtn, counterEl, nextBtn);
    mediaCol.appendChild(navBar);

    const thumbStrip = document.createElement("div");
    thumbStrip.className = "mo-gallery-thumbs";
    mediaCol.appendChild(thumbStrip);
    body.appendChild(mediaCol);

    // Right: triggers & params
    const infoCol = document.createElement("div");
    infoCol.className = "mo-gallery-info";

    const trigBlock = document.createElement("div");
    trigBlock.className = "mo-info-card";
    const trigHead = document.createElement("div");
    trigHead.className = "mo-info-card-head";
    const trigLabel = document.createElement("span");
    trigLabel.textContent = "模型触发词（Activation Text）";
    trigHead.appendChild(trigLabel);

    const trigInsertBtn = document.createElement("button");
    trigInsertBtn.type = "button";
    trigInsertBtn.className = "mo-btn mo-btn-mini";
    trigInsertBtn.textContent = "+ 塞入正向框";
    trigInsertBtn.disabled = !meta.activation_text;
    trigInsertBtn.addEventListener("click", () => {
      try {
        const added = appendPromptTokens(ctx.tabKind, meta.activation_text, false);
        showToast(added > 0 ? `已插入 ${added} 组触发词` : "触发词已在输入框中");
      } catch (err) {
        showToast(err.message, true);
      }
    });
    trigHead.appendChild(trigInsertBtn);
    trigBlock.appendChild(trigHead);

    const trigBox = document.createElement("div");
    trigBox.className = "mo-trigger-text";
    trigBox.textContent = meta.activation_text || "（当前 .json / .civitai.info 未记录触发词）";
    trigBlock.appendChild(trigBox);
    infoCol.appendChild(trigBlock);

    const paramPills = document.createElement("div");
    paramPills.className = "mo-param-pills";
    infoCol.appendChild(paramPills);

    const posLabel = document.createElement("div");
    posLabel.className = "mo-field-label";
    posLabel.textContent = "例图正向 Prompt";
    const posBox = document.createElement("textarea");
    posBox.className = "mo-prompt-box";
    posBox.readOnly = true;
    posBox.rows = 5;

    const negLabel = document.createElement("div");
    negLabel.className = "mo-field-label";
    negLabel.textContent = "例图负向 Negative Prompt";
    const negBox = document.createElement("textarea");
    negBox.className = "mo-prompt-box";
    negBox.readOnly = true;
    negBox.rows = 3;

    const actRow = document.createElement("div");
    actRow.className = "mo-gallery-actions";

    const fillBothBtn = document.createElement("button");
    fillBothBtn.type = "button";
    fillBothBtn.className = "mo-btn mo-btn-primary";
    fillBothBtn.textContent = "✨ 一键填入正负提示词";

    const appendPosBtn = document.createElement("button");
    appendPosBtn.type = "button";
    appendPosBtn.className = "mo-btn mo-btn-secondary";
    appendPosBtn.textContent = "➕ 追加正向词";

    const copyPosBtn = document.createElement("button");
    copyPosBtn.type = "button";
    copyPosBtn.className = "mo-btn mo-btn-secondary";
    copyPosBtn.textContent = "📋 复制 Prompt";

    actRow.append(fillBothBtn, appendPosBtn, copyPosBtn);
    infoCol.append(posLabel, posBox, negLabel, negBox, actRow);
    body.appendChild(infoCol);
    dialog.appendChild(body);
    overlay.appendChild(dialog);

    function renderSlide() {
      stage.innerHTML = "";
      paramPills.innerHTML = "";
      if (images.length === 0) {
        const empty = document.createElement("div");
        empty.className = "mo-gallery-empty";
        empty.textContent = meta.has_info
          ? "该版本的 .civitai.info 未附带作者静态例图。"
          : "未找到本地 .civitai.info（可在 Tab 2 → ③ 按 SHA-256 联网补全）。";
        stage.appendChild(empty);
        counterEl.textContent = "0 / 0";
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        posBox.value = "";
        negBox.value = "";
        fillBothBtn.disabled = true;
        appendPosBtn.disabled = true;
        copyPosBtn.disabled = true;
        return;
      }

      if (currentIdx < 0) currentIdx = images.length - 1;
      if (currentIdx >= images.length) currentIdx = 0;

      const cur = images[currentIdx];
      counterEl.textContent = `${currentIdx + 1} / ${images.length}`;
      prevBtn.disabled = images.length <= 1;
      nextBtn.disabled = images.length <= 1;

      if (isSafeHttpUrl(cur.url)) {
        const imgEl = document.createElement("img");
        imgEl.src = cur.url;
        imgEl.alt = `${meta.model_name || ctx.name} example ${currentIdx + 1}`;
        imgEl.loading = "eager";
        const shouldBlur = Boolean(cur.nsfw && configState.blur_nsfw_gallery && !revealed.has(currentIdx));
        if (shouldBlur) {
          imgEl.classList.add("mo-img-blurred");
        }
        stage.appendChild(imgEl);
        if (shouldBlur) {
          const veil = document.createElement("button");
          veil.type = "button";
          veil.className = "mo-nsfw-veil";
          veil.textContent = "🙈 NSFW 例图 · 点击查看";
          veil.addEventListener("click", () => {
            revealed.add(currentIdx);
            renderSlide();
          });
          stage.appendChild(veil);
        }
      }

      const pairs = [
        ["底模/大模型", cur.model],
        ["采样器", cur.sampler],
        ["步数", cur.steps],
        ["CFG", cur.cfg_scale],
        ["Seed", cur.seed],
        ["尺寸", cur.size],
      ];
      for (const [k, v] of pairs) {
        if (!v) continue;
        const p = document.createElement("span");
        p.className = "mo-param-pill";
        p.textContent = `${k}: ${v}`;
        paramPills.appendChild(p);
      }

      posBox.value = cur.prompt || "（作者未公开该例图 Prompt）";
      negBox.value = cur.negative_prompt || "";
      const hasPrompt = Boolean(cur.prompt);
      fillBothBtn.disabled = !hasPrompt;
      appendPosBtn.disabled = !hasPrompt;
      copyPosBtn.disabled = !hasPrompt;

      thumbStrip.querySelectorAll(".mo-gallery-thumb").forEach((t, idx) => {
        t.classList.toggle("active", idx === currentIdx);
      });
    }

    images.forEach((im, idx) => {
      if (!isSafeHttpUrl(im.url)) return;
      const tb = document.createElement("button");
      tb.type = "button";
      tb.className = "mo-gallery-thumb";
      const ti = document.createElement("img");
      ti.src = im.url;
      ti.alt = `thumb ${idx + 1}`;
      ti.loading = "lazy";
      if (im.nsfw && configState.blur_nsfw_gallery) {
        ti.classList.add("mo-thumb-blurred");
      }
      tb.appendChild(ti);
      tb.addEventListener("click", () => {
        currentIdx = idx;
        renderSlide();
      });
      thumbStrip.appendChild(tb);
    });

    prevBtn.addEventListener("click", () => {
      currentIdx -= 1;
      renderSlide();
    });
    nextBtn.addEventListener("click", () => {
      currentIdx += 1;
      renderSlide();
    });

    fillBothBtn.addEventListener("click", () => {
      const cur = images[currentIdx];
      if (!cur || !cur.prompt) return;
      try {
        replacePrompts(ctx.tabKind, cur.prompt, cur.negative_prompt || "");
        showToast("已将例图正负提示词填入当前绘图页");
      } catch (err) {
        showToast(err.message, true);
      }
    });

    appendPosBtn.addEventListener("click", () => {
      const cur = images[currentIdx];
      if (!cur || !cur.prompt) return;
      try {
        const added = appendPromptTokens(ctx.tabKind, cur.prompt, false);
        showToast(added > 0 ? `已追加 ${added} 段例图 Prompt` : "例图 Prompt 已在输入框中");
      } catch (err) {
        showToast(err.message, true);
      }
    });

    copyPosBtn.addEventListener("click", async () => {
      const cur = images[currentIdx];
      if (!cur || !cur.prompt) return;
      const ok = await copyText(cur.prompt);
      showToast(ok ? "已复制例图 Prompt 到剪贴板" : "复制失败，请手动选中复制", !ok);
    });

    overlay.addEventListener("click", (ev) => {
      if (ev.target === overlay) closeModal();
    });

    document.body.appendChild(overlay);
    activeModal = overlay;
    renderSlide();
  }

  async function handleAction(action, card) {
    const ctx = parseCardContext(card);
    if (!ctx) return;

    await refreshConfig(false);
    try {
      const meta = await fetchCardMeta(ctx);
      if (!meta || !meta.found) {
        showToast((meta && meta.error) || "未找到模型本地信息", true);
        return;
      }

      if (action === "open") {
        if (!isSafeHttpUrl(meta.civitai_url)) {
          showToast("缺少 .civitai.info 或 modelId（可到 Tab 2 → ③ 按 SHA-256 联网补全）", true);
          return;
        }
        window.open(meta.civitai_url, "_blank", "noopener,noreferrer");
        return;
      }

      if (action === "triggers") {
        if (!meta.activation_text) {
          showToast("该模型暂无可用触发词（.json 与 .civitai.info 均为空）", true);
          return;
        }
        const added = appendPromptTokens(ctx.tabKind, meta.activation_text, false);
        showToast(added > 0 ? `已插入触发词 (${added} 项)` : "触发词已在当前提示词框中");
        return;
      }

      if (action === "gallery") {
        openGalleryModal(meta, ctx);
      }
    } catch (err) {
      showToast(`读取模型信息失败：${err.message}`, true);
    }
  }

  function ensureCardToolbar(card) {
    if (!configState.enable_card_buttons) return;
    if (!card || card.querySelector(".mo-card-toolbar")) return;
    const ctx = parseCardContext(card);
    if (!ctx) return;

    const toolbar = document.createElement("div");
    toolbar.className = "mo-card-toolbar";

    const br = card.querySelector(".button-row");
    if (br && br.offsetHeight > 0) {
      toolbar.style.top = `${Math.max(28, br.offsetTop + br.offsetHeight + 4)}px`;
    }

    const defs = [
      { action: "open", icon: "🌐", title: "打开 C 站原页 (Model Organizer)" },
      { action: "triggers", icon: "🏷️", title: "一键塞入触发词到当前提示词框 (Model Organizer)" },
      { action: "gallery", icon: "🖼️", title: "查看作者例图与参考 Prompt (Model Organizer)" },
    ];

    for (const def of defs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "mo-card-btn";
      btn.dataset.moAction = def.action;
      btn.title = def.title;
      btn.setAttribute("aria-label", def.title);
      btn.textContent = def.icon;

      btn.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
      });
      btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        handleAction(def.action, card);
      });
      toolbar.appendChild(btn);
    }

    card.appendChild(toolbar);
  }

  document.addEventListener(
    "mouseover",
    (ev) => {
      if (!configState.enable_card_buttons) return;
      const target = ev.target;
      if (!(target instanceof Element)) return;
      const card = target.closest('[id$="_cards"] .card');
      if (card) {
        ensureCardToolbar(card);
      }
    },
    true
  );

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && activeModal) {
      closeModal();
    }
  });

  document.addEventListener("click", (ev) => {
    const target = ev.target;
    if (!(target instanceof Element)) return;
    if (target.closest('[id$="_extra_refresh"]') || target.closest("#settings_submit")) {
      metaCache.clear();
      setTimeout(() => refreshConfig(true), 400);
    }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => refreshConfig(true));
  } else {
    refreshConfig(true);
  }
})();
