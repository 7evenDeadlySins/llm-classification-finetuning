await(async () => {
  const MAX_ATTEMPTS = 999;  // 最大重试次数
  const AUTO_RUN_ALL = true; // 成功连上 G4 后自动点击“全部运行”

  window.__stopG4Retry = false;
  const runId = (window.__g4RetryRunId = (window.__g4RetryRunId || 0) + 1);
  const shouldStop = () => window.__stopG4Retry || window.__g4RetryRunId !== runId;

  class FatalError extends Error { }

  const SKIP_TAGS = new Set(['STYLE', 'SCRIPT', 'NOSCRIPT', 'TEMPLATE']);
  const RE = {
    g4: /(?:^|[^A-Za-z0-9])G4(?![A-Za-z0-9])/i,
    connectingBtn: /(?:正在连接|正在分配|正在初始化|正在重新连接|正在排队|\b(?:Connecting|Allocating|Initializing|Reconnecting|Queuing|Busy)\b)/i,
    connectingBar: /(?:正在连接|正在分配|正在初始化|正在重新连接|正在排队|\b(?:Connecting|Allocating|Initializing|Reconnecting|Queuing)\b)/i,
    disconnectedLabel: /^(?:连接|重新连接|(?:Connect|Reconnect)\b)/i,
    disconnectedTip: /点击即可连接|Click to connect/i,
    connectedTip: /已连接到|Connected to/i,
    connectedLabel: /已连接|磁盘|\b(?:Connected|Disk)\b/i,
    disconnectDlg: /断开连接并删除运行时|重置此运行时|Disconnect and delete runtime|reset this runtime/i,
    changeConfirm: /断开连接并删除运行时|重置此运行时|更改运行时类型|终止.*会话|删除.*运行时|Disconnect and delete runtime|reset this runtime|Change runtime type|terminate.*session/i,
    blocker: /验证|captcha|robot|配额|quota|usage limit|上限|超过/i,
    unavailable: /GPU|不可用|unavailable|CPU|无法连接|无法分配|连接失败|Cannot connect|Unable to connect|Could not connect/i,
    accelerator: /(?:G4|A100|L4|T4|H100|H200|B200|V100|TPU|CPU)/i,
  };

  const DLG_ACTIONS = {
    ok: {
      sels: ['md-filled-button[dialogaction="ok"]', '[dialogaction="ok"]'],
      btnSel: 'md-filled-button, md-text-button, md-outlined-button, button',
      re: /^(?:是|确定|确认|删除|保存|Yes|OK|Continue|Delete|Save)$/i,
    },
    cancel: {
      sels: ['[dialogaction="cancel"]', '[dialogaction="close"]'],
      btnSel: 'md-text-button, md-outlined-button, md-filled-button, button',
      re: /^(?:取消|不|关闭|知道了|Cancel|No|Close|Got it)$/i,
    },
  };

  // ---------- 1. 基础 DOM 与异步原语 ----------

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const waitFor = async (fn, timeout = 15_000, step = 150) => {
    const end = Number.isFinite(timeout) ? Date.now() + timeout : Infinity;
    while (Date.now() < end && !shouldStop()) {
      const val = fn();
      if (val) return val;
      await sleep(step);
    }
    return null;
  };

  const trySteps = async (steps, check, waitMs = 1200, stepMs = 80) => {
    for (const step of steps) {
      if (shouldStop() || check()) return Boolean(check());
      await step();
      if (await waitFor(check, waitMs, stepMs)) return true;
    }
    return Boolean(check());
  };

  const deepText = (root) => {
    const out = [];
    const walk = (node) => {
      if (!node) return;
      if (node.nodeType === 3) {
        const s = node.textContent.trim();
        if (s) out.push(s);
        return;
      }
      if (node.nodeType === 1 && (SKIP_TAGS.has(node.tagName) || (node.tagName === 'SLOT' && node.assignedNodes?.().length))) return;
      if (node.shadowRoot) walk(node.shadowRoot);
      for (const child of node.childNodes) walk(child);
    };
    walk(root);
    return out.join(' ').replace(/\s+/g, ' ').trim();
  };

  const deepQuery = (root, sel) => {
    if (!root) return null;
    let hit = root.querySelector(sel) || (root.shadowRoot && deepQuery(root.shadowRoot, sel));
    if (hit) return hit;
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot && (hit = deepQuery(el.shadowRoot, sel))) return hit;
    }
    return null;
  };

  const deepQueryFirst = (root, sels) => {
    for (const sel of sels) {
      const hit = deepQuery(root, sel);
      if (hit) return hit;
    }
    return null;
  };

  const deepQueryAll = (root, sel, out = []) => {
    if (!root) return out;
    out.push(...root.querySelectorAll(sel));
    if (root.shadowRoot) deepQueryAll(root.shadowRoot, sel, out);
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) deepQueryAll(el.shadowRoot, sel, out);
    }
    return out;
  };

  const hasBox = (el) => {
    if (!el?.isConnected) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };

  const isShown = (el) => {
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden' && Number(cs.opacity) > 0;
  };

  const isVisible = (el) => {
    if (!hasBox(el)) return false;
    for (let cur = el; cur?.nodeType === 1; cur = cur.parentElement || cur.getRootNode()?.host) {
      if (!isShown(cur)) return false;
    }
    return true;
  };

  const isBtnReady = (btn) => (
    hasBox(btn)
    && !btn.disabled
    && !btn.hasAttribute('disabled')
    && btn.getAttribute('data-aria-disabled') !== 'true'
    && isShown(btn)
  );

  const sendKey = (target, key, keyCode, types = ['keydown']) => {
    for (const type of types) {
      target.dispatchEvent(new KeyboardEvent(type, {
        key, code: key, keyCode, which: keyCode, bubbles: true, cancelable: true,
      }));
    }
  };

  const clickAt = async (target, { emitClick = true, emitPointer = true, hoverDelay = 0 } = {}) => {
    if (!target?.isConnected) return false;
    target.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    const { left, top, width, height } = target.getBoundingClientRect();
    const base = {
      bubbles: true, cancelable: true, composed: true, view: window,
      clientX: left + width / 2, clientY: top + height / 2, button: 0,
    };
    const emit = (mouseType, ptrType, buttons) => {
      if (ptrType && emitPointer && typeof PointerEvent === 'function') {
        target.dispatchEvent(new PointerEvent(ptrType, { ...base, buttons, pointerId: 1, pointerType: 'mouse' }));
      }
      target.dispatchEvent(new MouseEvent(mouseType, { ...base, buttons }));
    };

    emit('mouseover', 'pointerover', 0);
    emit('mousemove', null, 0);
    if (hoverDelay > 0) {
      await sleep(hoverDelay);
      if (!target.isConnected) return true;
    }
    emit('mousedown', 'pointerdown', 1);
    await sleep(35);
    if (!target.isConnected) return true;
    emit('mouseup', 'pointerup', 0);
    if (emitClick) emit('click', null, 0);
    return true;
  };

  const press = async (el) => {
    if (!el?.isConnected) return false;
    let target = el;
    while (target?.shadowRoot) {
      const next = target.shadowRoot.querySelector('md-text-button, md-filled-button, md-outlined-button, md-icon-button, button, input');
      if (!next) break;
      target = next;
    }
    if (target.tagName === 'BUTTON' || target.tagName === 'INPUT') {
      target.click();
      return true;
    }
    return clickAt(target);
  };

  // ---------- 2. Colab UI 状态查询 ----------

  const connectHost = () => document.querySelector('colab-connect-button');
  const statusBarEl = () => document.querySelector('colab-status-bar');
  const runtimeOptionsEl = () => deepQuery(statusBarEl(), '#runtime-options');
  const statusText = () => deepText(runtimeOptionsEl());
  const fallbackSnackbar = () => document.querySelector('colab-snackbar.gpu-model-fallback[open]');

  const connectInfo = () => {
    const btn = deepQuery(connectHost(), '#connect');
    return {
      btn,
      txt: deepText(deepQuery(btn, '.connect-button-content') || btn),
      tip: btn?.getAttribute('tooltiptext') || '',
    };
  };

  const openDialogs = (excludeChange = true, excludeOpening = false) => [
    ...document.querySelectorAll(
      `md-dialog[open]${excludeChange ? ':not(.change-runtime-type)' : ''}${excludeOpening ? ':not(.colab-dialog-opening)' : ''}`,
    ),
  ];

  const topDialog = (excludeChange = true, excludeOpening = false) => openDialogs(excludeChange, excludeOpening).pop() || null;
  const isDialogClosed = (dlg) => !dlg || !dlg.isConnected || !dlg.hasAttribute('open');
  const isDisconnectDialog = (dlg) => Boolean(dlg && !dlg.classList.contains('change-runtime-type') && RE.disconnectDlg.test(deepText(dlg)));

  const isConnecting = () => {
    const { txt, tip } = connectInfo();
    return RE.connectingBtn.test(txt) || RE.connectingBtn.test(tip) || RE.connectingBar.test(deepText(statusBarEl()));
  };

  const isDisconnected = () => {
    if (isConnecting() || (runtimeOptionsEl() && statusText())) return false;
    const { txt, tip } = connectInfo();
    return RE.disconnectedLabel.test(txt) || RE.disconnectedTip.test(tip);
  };

  const isConnected = () => {
    if (isDisconnected() || isConnecting()) return false;
    const st = statusText();
    if (runtimeOptionsEl() && st && !RE.connectingBar.test(st)) return true;
    const { txt, tip } = connectInfo();
    return RE.connectedTip.test(tip) || RE.connectedLabel.test(txt);
  };

  const isActiveConnect = () => isConnecting() || isConnected() || openDialogs().some((d) => !isDisconnectDialog(d));

  const getBlocker = () => {
    const frames = document.querySelectorAll('iframe[src*="recaptcha"], iframe[title*="challenge" i], iframe[name^="c-"], iframe[title*="reCAPTCHA"]');
    if ([...frames].some(isVisible)) return '检测到可见的 reCAPTCHA 人机验证';
    for (const dlg of openDialogs()) {
      if (isDisconnectDialog(dlg)) continue;
      const txt = deepText(dlg);
      if (RE.blocker.test(txt)) return `检测到阻断弹窗: ${txt.slice(0, 80)}`;
    }
    return null;
  };

  // ---------- 3. 弹窗与菜单控件操作 ----------

  const waitDialogsSettled = async (timeout = 1200) => {
    await waitFor(() => !document.querySelector('md-dialog.colab-dialog-opening, md-dialog.colab-dialog-closing'), timeout, 60);
    await sleep(80);
  };

  const waitDialogReady = async (selector, timeout = 5000) => {
    const dlg = await waitFor(() => {
      const d = document.querySelector(selector);
      return d && !d.classList.contains('colab-dialog-opening') ? d : null;
    }, timeout, 80);
    if (dlg) await sleep(120);
    return dlg;
  };

  const findDialogBtn = (dlg, action = 'ok') => {
    if (!dlg) return null;
    const { sels, btnSel, re } = DLG_ACTIONS[action];
    return deepQueryFirst(dlg, sels) || deepQueryAll(dlg, btnSel).find((x) => re.test(deepText(x))) || null;
  };

  const confirmDialog = async (dlg, action = 'ok') => {
    const closed = () => isDialogClosed(dlg);
    if (closed()) return true;
    await waitFor(() => closed() || !dlg.classList.contains('colab-dialog-opening'), 2500, 80);
    if (closed()) return true;
    await sleep(100);

    const btn = await waitFor(() => {
      if (closed()) return { closed: true };
      const b = findDialogBtn(dlg, action);
      return isBtnReady(b) ? b : null;
    }, 3000, 80);

    if (btn?.closed || closed()) return true;
    if (!btn) {
      if (action !== 'cancel') return false;
      return trySteps([() => dlg.close?.('cancel'), () => sendKey(document, 'Escape', 27)], closed, 1500);
    }

    return trySteps([
      () => press(btn),
      async () => {
        btn.click();
        await sleep(80);
        if (!closed()) await clickAt(btn);
      },
      () => dlg.close?.(action === 'ok' ? (btn.getAttribute('value') || 'ok') : 'cancel'),
      ...(action === 'cancel' ? [() => sendKey(document, 'Escape', 27)] : []),
    ], closed, 1200);
  };

  const findVisibleMenuItem = (cmd) => {
    const sel = `[role="menuitem"][command="${cmd}"]`;
    return [...document.querySelectorAll(sel)].find(isVisible) || deepQueryAll(document, sel).find(isVisible) || null;
  };

  const isMenuExpanded = (menuBtn, cmd = null) => Boolean(
    menuBtn && ((menuBtn.getAttribute('aria-expanded') === 'true' && isVisible(document.querySelector('#runtime-menu'))) || (cmd && findVisibleMenuItem(cmd))),
  );

  const closeRuntimeMenu = async (menuBtn) => {
    if (!menuBtn) return;
    const isClosed = () => menuBtn.getAttribute('aria-expanded') !== 'true' && !isVisible(document.querySelector('#runtime-menu'));
    if (isClosed() && document.activeElement !== menuBtn) return;
    if (!isClosed()) {
      await trySteps([
        () => sendKey(document, 'Escape', 27),
        () => clickAt(menuBtn, { emitClick: false, emitPointer: false }),
      ], isClosed, 800);
    }
    menuBtn.blur?.();
    await sleep(60);
  };

  const clickMenuItem = async (cmd, menuBtn = null) => {
    const item = await waitFor(() => findVisibleMenuItem(cmd), 3000, 80);
    if (!item) return null;
    await sleep(120);
    if (!item.isConnected || !isVisible(item)) return null;
    if (item.getAttribute('aria-disabled') === 'true' || item.classList.contains('goog-menuitem-disabled')) {
      await closeRuntimeMenu(menuBtn);
      return false;
    }

    const dialogsBefore = document.querySelectorAll('md-dialog[open]').length;
    const isClosed = () => !(
      (!menuBtn || menuBtn.getAttribute('aria-expanded') === 'true')
      && isVisible(item)
      && document.querySelectorAll('md-dialog[open]').length === dialogsBefore
    );

    await trySteps([
      () => clickAt(deepQuery(item, '.goog-menuitem-content') || item, { emitClick: false, emitPointer: false, hoverDelay: 60 }),
      () => item.click(),
      () => sendKey(item.closest('[role="menu"]') || item, 'Enter', 13, ['keydown', 'keyup']),
    ], isClosed, 450);
    return true;
  };

  const runMenuCommand = async (cmd) => {
    const menuBtn = document.querySelector('#runtime-menu-button');
    if (!menuBtn) throw new FatalError('未找到 #runtime-menu-button');

    await waitDialogsSettled();
    if (shouldStop()) return false;

    const isOpen = () => isMenuExpanded(menuBtn, cmd);
    if (!isOpen()) {
      if (document.activeElement === menuBtn) await closeRuntimeMenu(menuBtn);
      await trySteps([
        () => clickAt(menuBtn, { emitClick: false, emitPointer: false }),
        async () => { await closeRuntimeMenu(menuBtn); menuBtn.click(); },
        async () => { await closeRuntimeMenu(menuBtn); menuBtn.focus?.(); sendKey(menuBtn, 'ArrowDown', 40); },
      ], isOpen, 1200);
    }

    if (shouldStop()) return false;
    if (!isOpen()) {
      await closeRuntimeMenu(menuBtn);
      throw new Error(`Runtime 菜单未能展开: ${cmd}`);
    }

    const clicked = await clickMenuItem(cmd, menuBtn);
    if (clicked === null) {
      await closeRuntimeMenu(menuBtn);
      throw new Error(`菜单内未找到可见菜单项: ${cmd}`);
    }
    return clicked;
  };

  const dismissFallback = async () => {
    const toast = fallbackSnackbar();
    if (!toast) return;
    const closeBtn = deepQueryFirst(toast, [
      'md-icon-button.close',
      'md-icon-button[aria-label*="close" i], md-icon-button[aria-label*="关闭"]',
      'md-icon-button, button',
    ]);
    await trySteps([
      ...(closeBtn ? [() => press(closeBtn)] : []),
      () => (typeof toast.close === 'function' ? toast.close() : toast.removeAttribute('open')),
    ], () => !fallbackSnackbar(), 1000);
  };

  // ---------- 4. 运行时业务动作 ----------

  const clearDisconnectDialogs = async () => {
    for (let i = 0; i < 5 && !shouldStop(); i++) {
      const dlg = openDialogs().filter(isDisconnectDialog).pop();
      if (!dlg) break;
      await confirmDialog(dlg, 'cancel');
    }
  };

  const triggerConnect = async () => {
    if (isActiveConnect()) return true;
    const { btn } = connectInfo();
    const content = btn && deepQuery(btn, '.connect-button-content');
    if (await trySteps([
      ...(btn ? [() => press(btn)] : []),
      ...(content ? [() => clickAt(content)] : []),
    ], isActiveConnect, 2000)) return true;

    const dd = deepQuery(connectHost(), '#connect-dropdown');
    if (!dd) return false;
    console.log('[G4] 直接点击未触发，改用连接下拉菜单 connect-managed-gce-vm...');
    await trySteps([() => press(dd), () => clickAt(dd)], () => findVisibleMenuItem('connect-managed-gce-vm'), 1200);
    return Boolean(await clickMenuItem('connect-managed-gce-vm', dd) && await waitFor(isActiveConnect, 3000, 80));
  };

  const releaseRuntime = async () => {
    if (isDisconnected()) {
      await clearDisconnectDialogs();
      return true;
    }
    const existing = openDialogs().filter(isDisconnectDialog).pop();
    if (existing) {
      await confirmDialog(existing, 'ok');
    } else {
      if (!(await runMenuCommand('powerwash-current-vm'))) {
        await clearDisconnectDialogs();
        return Boolean(await waitFor(isDisconnected, 3000, 80));
      }
      const res = await waitFor(() => (isDisconnected() ? { done: true } : topDialog() && { dlg: topDialog() }), 5000, 80);
      if (res?.dlg) await confirmDialog(res.dlg, 'ok');
    }
    const ok = await waitFor(isDisconnected, 15_000, 150);
    await clearDisconnectDialogs();
    return Boolean(ok || isDisconnected());
  };

  const switchToG4 = async ({ forceDialog = false } = {}) => {
    if (isConnecting()) return true;
    const wasConnected = isConnected();
    if (!forceDialog && !wasConnected && RE.g4.test(connectInfo().txt) && !document.querySelector('md-dialog.change-runtime-type[open]')) {
      return true;
    }

    await dismissFallback();

    let dlg = document.querySelector('md-dialog.change-runtime-type[open]');
    for (let attempt = 0; attempt < 2 && !dlg && !shouldStop(); attempt++) {
      if (isConnecting()) return true;
      if ((await runMenuCommand('change-runtime-type')) === false) {
        await sleep(400);
        return isConnecting();
      }
      if (shouldStop()) return false;
      dlg = await waitDialogReady('md-dialog.change-runtime-type[open]', 3000);
      if (!dlg && isConnecting()) return true;
    }
    if (shouldStop()) return false;
    if (!dlg) throw new Error('未能打开“更改运行时类型”弹窗');

    const radio = await waitFor(() => deepQuery(dlg, 'mwc-radio[name="accelerator"][value="GPU,G4"], md-radio[name="accelerator"][value="GPU,G4"]'), 3000, 80);
    if (!radio || radio.disabled || radio.hasAttribute('disabled')) {
      await confirmDialog(dlg, 'cancel');
      throw new FatalError(radio ? 'G4 GPU 选项当前被禁用' : '弹窗内未找到 G4 GPU 选项');
    }

    const checked = () => (typeof radio.checked === 'boolean' ? radio.checked : radio.getAttribute('aria-checked') === 'true');
    if (!checked()) {
      await press(radio);
      await waitFor(checked, 1500, 50);
    }

    if (!isBtnReady(findDialogBtn(dlg, 'ok'))) {
      await confirmDialog(dlg, 'cancel');
      await waitDialogsSettled();
      if (!wasConnected) return true;
      console.warn('[G4] 保存按钮不可用，兜底释放当前非 G4 运行时...');
      await releaseRuntime();
      return false;
    }

    await confirmDialog(dlg, 'ok');

    if (!wasConnected) {
      await waitFor(() => RE.g4.test(connectInfo().txt) || isConnecting() || isConnected(), 3000, 80);
      return true;
    }

    const switched = () => isConnecting() || isDisconnected() || RE.g4.test(statusText()) || Boolean(getBlocker());
    const transition = await waitFor(() => (switched() ? { done: true } : topDialog() && { dlg: topDialog() }), 10_000, 80);
    if (transition?.dlg && !getBlocker() && RE.changeConfirm.test(deepText(transition.dlg))) {
      await confirmDialog(transition.dlg, 'ok');
      await waitFor(switched, 10_000, 80);
    }

    if (isConnected() && !isConnecting() && !RE.g4.test(statusText()) && !getBlocker()) {
      console.warn('[G4] 保存 G4 后旧非 G4 运行时未自动切换，兜底释放...');
      await releaseRuntime();
      return false;
    }

    if (isDisconnected() && !isConnecting() && !topDialog(true, true) && !getBlocker()) {
      await waitFor(() => isConnecting() || isConnected() || Boolean(topDialog(true, true)) || Boolean(getBlocker()), 2000, 80);
    }
    return true;
  };

  const playSuccessSound = () => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      const ctx = new Ctx();
      ctx.resume?.().catch(() => { });
      [523.25, 659.25, 783.99].forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        const t = ctx.currentTime + idx * 0.12;
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, t);
        gain.gain.setValueAtTime(0.2, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(t);
        osc.stop(t + 0.25);
      });
      setTimeout(() => ctx.close?.(), 800);
    } catch { }
  };

  const completeIfG4 = async (st) => {
    if (!RE.g4.test(st) || fallbackSnackbar()) return false;
    console.log('[G4] 成功连接 G4:', st);
    playSuccessSound();
    if (AUTO_RUN_ALL) {
      const runBtn = document.querySelector('#toolbar-run-button');
      if (!runBtn || !(await press(runBtn))) {
        console.warn('[G4] 未能触发全部运行按钮 (#toolbar-run-button)');
      }
    }
    return true;
  };

  // ---------- 5. 主流程控制 ----------

  const startAttempt = async (attempt) => {
    if (isDisconnected()) await clearDisconnectDialogs();

    if (isConnecting()) {
      console.log(`[G4] 第 ${attempt}/${MAX_ATTEMPTS} 轮：检测到正在连接/分配，继续等待...`);
      return 'wait';
    }

    if (isConnected()) {
      const st = statusText() || await waitFor(() => statusText() || null, 2000, 80) || '';
      if (await completeIfG4(st)) return 'done';
      console.log(`[G4] 第 ${attempt}/${MAX_ATTEMPTS} 轮：当前为非 G4 (${st || '未知'})，直接选择 G4 保存以重连...`);
      if (!(await switchToG4({ forceDialog: true }))) return 'retry';
      if (isDisconnected() && !isConnecting() && !topDialog(true, true) && !getBlocker() && !(await triggerConnect())) {
        console.warn('[G4] 保存后未自动重连且触发连接失败，立即重试...');
        return 'retry';
      }
      return 'wait';
    }

    if (!(await switchToG4({ forceDialog: false }))) return 'retry';
    console.log(`[G4] 第 ${attempt}/${MAX_ATTEMPTS} 次尝试连接 G4...`);
    if (!isActiveConnect() && !(await triggerConnect())) {
      console.warn('[G4] 触发连接失败，立即重试...');
      return 'retry';
    }
    return 'wait';
  };

  const waitOutcome = () => {
    let disconnectedAt = 0;
    return waitFor(() => {
      const blocker = getBlocker();
      if (blocker) return { stop: blocker };

      const dlg = topDialog(true, true);
      if (dlg && !isDisconnectDialog(dlg)) {
        disconnectedAt = 0;
        if (RE.unavailable.test(deepText(dlg))) return { dlg };
      }

      const fb = fallbackSnackbar();
      if (fb && (isConnected() || (statusText() && !RE.g4.test(statusText())))) {
        return { fallback: deepText(fb) };
      }

      if (isConnected()) return { connected: true };

      if (isDisconnected() && !isConnecting()) {
        if (!disconnectedAt) disconnectedAt = Date.now();
        if (Date.now() - disconnectedAt >= 3000) return { disconnected: true };
      } else {
        disconnectedAt = 0;
      }
      return null;
    }, Infinity, 150);
  };

  const handleOutcome = async (outcome) => {
    if (outcome.stop) {
      console.warn('[G4]', outcome.stop);
      return true;
    }
    if (outcome.dlg) {
      await confirmDialog(outcome.dlg, 'cancel');
      console.log('[G4] G4 不可用（弹窗提示），立即重试');
      return false;
    }
    if (outcome.disconnected) {
      console.warn('[G4] 连接已中断并回到未连接状态，立即重试...');
      return false;
    }
    if (outcome.fallback) {
      console.log(`[G4] 检测到回退 (${statusText() || outcome.fallback})，立即切换 G4 并保存...`);
      return false;
    }

    const st = (await waitFor(() => {
      if (fallbackSnackbar()) return '__FALLBACK__';
      const s = statusText();
      return RE.accelerator.test(s) ? s : null;
    }, 5000, 80)) || statusText();

    if (shouldStop()) {
      console.log('[G4] 已手动停止');
      return true;
    }
    if (st !== '__FALLBACK__' && await completeIfG4(st)) return true;

    const label = (st && st !== '__FALLBACK__' ? st : '') || deepText(fallbackSnackbar()) || '未知';
    console.log(`[G4] 回退为非 G4 (${label})，立即切换 G4 并保存...`);
    return false;
  };

  const recover = async () => {
    const openChangeDlg = document.querySelector('md-dialog.change-runtime-type[open]');
    if (openChangeDlg) await confirmDialog(openChangeDlg, 'cancel');
    await closeRuntimeMenu(document.querySelector('#runtime-menu-button'));
    await waitDialogsSettled();
  };

  let finished = false;
  for (let i = 1; i <= MAX_ATTEMPTS && !shouldStop(); i++) {
    const blocker = getBlocker();
    if (blocker) {
      console.warn('[G4]', blocker);
      finished = true;
      break;
    }

    try {
      const state = await startAttempt(i);
      if (state === 'done') {
        finished = true;
        break;
      }
      if (state === 'retry') continue;

      const outcome = await waitOutcome();
      if (shouldStop()) {
        console.log('[G4] 已手动停止');
        finished = true;
        break;
      }
      if (!outcome || await handleOutcome(outcome)) {
        finished = Boolean(outcome);
        break;
      }
    } catch (err) {
      if (err instanceof FatalError) {
        console.error('[G4] 终止:', err.message);
        finished = true;
        break;
      }
      console.warn('[G4] 本轮异常，立即重试:', err?.message || err);
      await recover();
    }
  }

  if (!finished && !shouldStop()) {
    console.warn(`[G4] 已达到最大重试次数 (${MAX_ATTEMPTS})，停止重试`);
  }
})();