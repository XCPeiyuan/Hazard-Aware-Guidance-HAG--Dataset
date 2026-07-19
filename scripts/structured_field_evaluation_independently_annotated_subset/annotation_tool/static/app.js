const $ = (id) => document.getElementById(id);
const requiredElement = (id) => {
  const element = $(id);
  if (!element) throw new Error(`界面文件版本不一致，缺少元素 #${id}。请按 Ctrl+F5 强制刷新。`);
  return element;
};
const apiPath = (path) => path.split("/").map(encodeURIComponent).join("/");
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const uid = () => crypto.randomUUID ? crypto.randomUUID() : `h-${Date.now()}-${Math.random().toString(16).slice(2)}`;

const state = {
  images: [],
  summary: { total: 0, completed: 0, draft: 0, unannotated: 0, role_completed: 0, role_pending: 0 },
  annotation: null,
  currentPath: null,
  selectedId: null,
  filter: "all",
  roleFilter: "all",
  tool: "select",
  zoom: 1,
  dirty: false,
  saveTimer: null,
  gesture: null,
  suggestions: [],
  spaceDown: false,
};

const directionLabels = {
  right: "右侧", right_ahead: "右前方", slightly_right_ahead: "略偏右前",
  ahead: "正前方", slightly_left_ahead: "略偏左前", left_ahead: "左前方", left: "左侧",
};
const roleLabels = { required: "必须", optional: "可选", ignore: "忽略" };

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).detail || message; } catch {}
    throw new Error(message);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response;
}

function toast(message, error = false) {
  const el = $("toast");
  el.textContent = message;
  el.className = `toast show${error ? " error" : ""}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.className = "toast", 2200);
}

function markDirty() {
  if (!state.annotation) return;
  state.dirty = true;
  $("save-state").textContent = "有未保存修改";
  $("save-state").className = "save-state dirty";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => saveAnnotation(false), 900);
}

async function refreshImages() {
  const data = await request("/api/images");
  state.images = data.images;
  state.summary = data.summary;
  state.suggestions = data.object_name_suggestions;
  renderProgress();
  renderImageList();
  renderSuggestions();
  return data.state.last_opened_image;
}

function renderProgress() {
  const { total, completed, draft, unannotated, role_completed } = state.summary;
  $("progress-text").textContent = `${completed} / ${total}`;
  $("progress-fill").style.width = total ? `${completed / total * 100}%` : "0%";
  $("role-progress-text").textContent = `${role_completed} / ${total}`;
  $("role-progress-fill").style.width = total ? `${role_completed / total * 100}%` : "0%";
  $("draft-count").textContent = draft;
  $("unannotated-count").textContent = unannotated;
}

function filteredImages() {
  return state.images.filter(image =>
    (state.filter === "all" || image.status === state.filter) &&
    (state.roleFilter === "all" || image.evaluation_review_status === state.roleFilter)
  );
}

function renderImageList() {
  const list = $("image-list");
  list.innerHTML = "";
  for (const image of filteredImages()) {
    const item = document.createElement("button");
    item.className = `image-item${image.path === state.currentPath ? " active" : ""}`;
    item.innerHTML = `
      <img src="/api/image/${apiPath(image.path)}" loading="lazy" alt="">
      <span><span class="image-name">${escapeHtml(image.name)}</span>
      <span class="image-meta">${image.hazard_count} 个障碍 · ${image.status === "completed" ? "标注完成" : image.status === "draft" ? "草稿" : "未标注"} · ${image.evaluation_review_status === "completed" ? "角色已检查" : "角色待检查"}</span></span>
      <span class="status-markers"><span class="status-dot ${image.status}"></span><span class="role-dot ${image.evaluation_review_status}"></span></span>`;
    item.onclick = () => loadImage(image.path);
    list.appendChild(item);
  }
}

function renderSuggestions() {
  $("object-name-options").innerHTML = state.suggestions.map(value => `<option value="${escapeHtml(value)}"></option>`).join("");
}

async function loadImage(path) {
  if (!path || path === state.currentPath) return;
  await saveAnnotation(false);
  state.currentPath = path;
  state.selectedId = null;
  state.annotation = await request(`/api/annotations/${apiPath(path)}`);
  state.dirty = false;
  $("current-path").textContent = path;
  $("main-image").src = `/api/image/${apiPath(path)}`;
  $("main-image").onload = () => fitImage();
  $("image-stage").hidden = false;
  $("empty-canvas").hidden = true;
  $("image-notes").value = state.annotation.image_notes || "";
  $("save-state").textContent = state.annotation.updated_at ? "已保存" : "新图片";
  $("save-state").className = "save-state saved";
  await request(`/api/state/last-opened/${apiPath(path)}`, { method: "POST" });
  renderAll();
}

function renderAll() {
  renderImageList();
  renderHazards();
  renderBoxes();
  renderEditor();
  updateButtons();
}

function renderHazards() {
  const hazards = state.annotation?.hazards || [];
  $("hazard-count").textContent = `${hazards.length} 个`;
  $("hazard-list").innerHTML = hazards.map((hazard, index) => `
    <button class="hazard-item ${hazard.id === state.selectedId ? "active" : ""}" data-id="${hazard.id}">
      <span><strong>${index + 1}. ${escapeHtml(hazard.object_name || "未命名障碍")}</strong><span class="role-badge ${hazard.evaluation_role || ""}">${roleLabels[hazard.evaluation_role] || "未检查"}</span><br>
      <small>${hazard.hazard_category || "未选类别"} · ${hazard.distance_steps || "?"} 步 · ${directionLabels[hazard.direction_bin] || "未选方向"}</small></span>
    </button>`).join("");
  document.querySelectorAll(".hazard-item").forEach(item => item.onclick = () => selectHazard(item.dataset.id));
}

function renderBoxes() {
  const overlay = $("overlay");
  overlay.innerHTML = "";
  if (!state.annotation) return;
  state.annotation.hazards.forEach((hazard, index) => {
    const box = hazard.bbox;
    if (!box) return;
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.dataset.id = hazard.id;
    group.innerHTML = `
      <rect class="box ${hazard.id === state.selectedId ? "selected" : ""}" x="${box.x * 100}%" y="${box.y * 100}%" width="${box.width * 100}%" height="${box.height * 100}%"></rect>
      <text class="box-label" x="${box.x * 100}%" y="${Math.max(box.y * 100 - 0.6, 1.5)}%">${index + 1} ${escapeHtml(hazard.object_name || "")}</text>
      ${hazard.id === state.selectedId ? `<rect class="resize-handle" data-handle="resize" x="${(box.x + box.width) * 100}%" y="${(box.y + box.height) * 100}%" width="12" height="12" transform="translate(-6 -6)"></rect>` : ""}`;
    group.querySelector(".box").addEventListener("pointerdown", startBoxMove);
    const handle = group.querySelector(".resize-handle");
    if (handle) handle.addEventListener("pointerdown", startBoxResize);
    overlay.appendChild(group);
  });
}

function selectedHazard() {
  return state.annotation?.hazards.find(h => h.id === state.selectedId) || null;
}

function selectHazard(id) {
  state.selectedId = id;
  state.tool = "select";
  renderAll();
}

function renderEditor() {
  const hazard = selectedHazard();
  $("no-selection").hidden = Boolean(hazard);
  $("editor-fields").hidden = !hazard;
  if (!hazard) return;
  $("object-name").value = hazard.object_name || "";
  $("distance-steps").value = hazard.distance_steps || "";
  $("hazard-notes").value = hazard.notes || "";
  $("role-notes").value = hazard.role_notes || "";
  document.querySelectorAll("[name=category]").forEach(input => input.checked = input.value === hazard.hazard_category);
  document.querySelectorAll("[name=direction]").forEach(input => input.checked = input.value === hazard.direction_bin);
  document.querySelectorAll("[name=evaluation-role]").forEach(input => input.checked = input.value === hazard.evaluation_role);
  $("bbox-readout").textContent = JSON.stringify(hazard.bbox, null, 2);
}

function updateButtons() {
  const hasSelection = Boolean(selectedHazard());
  $("duplicate-btn").disabled = !hasSelection;
  $("delete-btn").disabled = !hasSelection;
  $("complete-btn").textContent = state.annotation?.status === "completed" ? "已完成" : "标记完成";
  $("complete-btn").disabled = !state.annotation || state.annotation.status === "completed";
  $("complete-role-review-btn").textContent = state.annotation?.evaluation_review_status === "completed" ? "角色已检查" : "完成角色检查";
  $("complete-role-review-btn").disabled = !state.annotation || state.annotation.status !== "completed" || state.annotation.evaluation_review_status === "completed";
  $("select-tool").classList.toggle("active", state.tool === "select");
  $("draw-tool").classList.toggle("active", state.tool === "draw");
}

function addHazard(copy = null) {
  if (!state.annotation) return;
  const base = copy || {};
  const hazard = {
    id: uid(),
    bbox: copy?.bbox ? {
      x: clamp(copy.bbox.x + .02, 0, .85), y: clamp(copy.bbox.y + .02, 0, .85),
      width: copy.bbox.width, height: copy.bbox.height,
    } : null,
    object_name: base.object_name || "",
    hazard_category: base.hazard_category || "",
    distance_steps: base.distance_steps || null,
    direction_bin: base.direction_bin || "",
    notes: base.notes || "",
    evaluation_role: copy ? base.evaluation_role || "" : "",
    role_notes: copy ? base.role_notes || "" : "",
  };
  state.annotation.hazards.push(hazard);
  state.selectedId = hazard.id;
  state.annotation.status = "draft";
  state.annotation.evaluation_review_status = "pending";
  state.tool = hazard.bbox ? "select" : "draw";
  markDirty();
  renderAll();
  if (!hazard.bbox) toast("请在图片上拖动绘制障碍框");
}

function deleteSelected() {
  if (!state.annotation || !state.selectedId) return;
  state.annotation.hazards = state.annotation.hazards.filter(h => h.id !== state.selectedId);
  state.selectedId = state.annotation.hazards[0]?.id || null;
  state.annotation.status = "draft";
  state.annotation.evaluation_review_status = "pending";
  markDirty();
  renderAll();
}

function stagePoint(event) {
  const rect = $("image-stage").getBoundingClientRect();
  return { x: clamp((event.clientX - rect.left) / rect.width, 0, 1), y: clamp((event.clientY - rect.top) / rect.height, 0, 1) };
}

function startDraw(event) {
  if (state.spaceDown || state.tool !== "draw" || event.target !== $("overlay")) return;
  event.preventDefault();
  let hazard = selectedHazard();
  if (!hazard || hazard.bbox) {
    addHazard();
    hazard = selectedHazard();
  }
  const start = stagePoint(event);
  hazard.bbox = { x: start.x, y: start.y, width: .001, height: .001 };
  state.gesture = { type: "draw", start, hazard };
  $("overlay").setPointerCapture(event.pointerId);
  renderBoxes();
}

function startBoxMove(event) {
  if (state.spaceDown || state.tool !== "select") return;
  event.stopPropagation();
  const id = event.currentTarget.parentElement.dataset.id;
  selectHazard(id);
  const hazard = selectedHazard();
  state.gesture = { type: "move", start: stagePoint(event), original: { ...hazard.bbox }, hazard };
  $("overlay").setPointerCapture(event.pointerId);
}

function startBoxResize(event) {
  event.stopPropagation();
  const id = event.currentTarget.parentElement.dataset.id;
  selectHazard(id);
  const hazard = selectedHazard();
  state.gesture = { type: "resize", start: stagePoint(event), original: { ...hazard.bbox }, hazard };
  $("overlay").setPointerCapture(event.pointerId);
}

function moveGesture(event) {
  if (!state.gesture) return;
  const point = stagePoint(event);
  const { type, start, original, hazard } = state.gesture;
  if (type === "draw") {
    hazard.bbox = {
      x: Math.min(start.x, point.x), y: Math.min(start.y, point.y),
      width: Math.max(Math.abs(point.x - start.x), .001), height: Math.max(Math.abs(point.y - start.y), .001),
    };
  } else if (type === "move") {
    hazard.bbox.x = clamp(original.x + point.x - start.x, 0, 1 - original.width);
    hazard.bbox.y = clamp(original.y + point.y - start.y, 0, 1 - original.height);
  } else if (type === "resize") {
    hazard.bbox.width = clamp(original.width + point.x - start.x, .005, 1 - original.x);
    hazard.bbox.height = clamp(original.height + point.y - start.y, .005, 1 - original.y);
  }
  renderBoxes();
}

function endGesture() {
  if (!state.gesture) return;
  const hazard = state.gesture.hazard;
  if (hazard.bbox.width < .005 || hazard.bbox.height < .005) hazard.bbox = null;
  state.gesture = null;
  state.tool = "select";
  state.annotation.status = "draft";
  markDirty();
  renderAll();
}

function fitImage() {
  if (!state.annotation) return;
  const viewport = $("canvas-viewport");
  const padding = 54;
  state.zoom = Math.min((viewport.clientWidth - padding) / state.annotation.image_width, (viewport.clientHeight - padding) / state.annotation.image_height, 1);
  applyZoom();
}

function applyZoom() {
  if (!state.annotation) return;
  const stage = $("image-stage");
  stage.style.width = `${state.annotation.image_width * state.zoom}px`;
  stage.style.height = `${state.annotation.image_height * state.zoom}px`;
  $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
}

function changeZoom(factor) {
  state.zoom = clamp(state.zoom * factor, .05, 4);
  applyZoom();
}

async function saveAnnotation(notify = true) {
  clearTimeout(state.saveTimer);
  if (!state.annotation || !state.currentPath || !state.dirty) return;
  try {
    state.annotation = await request(`/api/annotations/${apiPath(state.currentPath)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.annotation),
    });
    state.dirty = false;
    $("save-state").textContent = "已保存";
    $("save-state").className = "save-state saved";
    await refreshImages();
    if (notify) toast("草稿已保存");
  } catch (error) {
    $("save-state").textContent = "保存失败";
    toast(error.message, true);
  }
}

async function completeAnnotation() {
  if (!state.annotation) return;
  try {
    const payload = { ...state.annotation, status: "completed" };
    state.annotation = await request(`/api/annotations/${apiPath(state.currentPath)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    state.dirty = false;
    $("save-state").textContent = "已完成并保存";
    $("save-state").className = "save-state saved";
    await refreshImages();
    renderAll();
    toast("当前图片已标记完成");
  } catch (error) { toast(`无法完成：${error.message}`, true); }
}

async function completeRoleReview() {
  if (!state.annotation || state.annotation.status !== "completed") return;
  if (state.annotation.hazards.some(hazard => !roleLabels[hazard.evaluation_role])) {
    toast("请先为每个障碍物选择评估角色", true);
    return;
  }
  state.annotation.evaluation_review_status = "completed";
  state.dirty = true;
  await saveAnnotation(false);
  if (!state.dirty) {
    renderAll();
    toast("角色检查已完成；匹配镜像会自动继承");
  }
}

function navigate(offset) {
  const list = filteredImages();
  if (!list.length) return;
  const index = Math.max(0, list.findIndex(img => img.path === state.currentPath));
  loadImage(list[clamp(index + offset, 0, list.length - 1)].path);
}

function nextUnfinished() {
  const start = Math.max(0, state.images.findIndex(img => img.path === state.currentPath));
  const ordered = [...state.images.slice(start + 1), ...state.images.slice(0, start + 1)];
  const target = ordered.find(img => img.status !== "completed");
  if (target) loadImage(target.path); else toast("全部图片均已完成");
}

function nextRolePending() {
  const start = Math.max(0, state.images.findIndex(img => img.path === state.currentPath));
  const ordered = [...state.images.slice(start + 1), ...state.images.slice(0, start + 1)];
  const target = ordered.find(img => img.evaluation_review_status !== "completed");
  if (target) loadImage(target.path); else toast("全部图片均已完成角色检查");
}

function bindEditor() {
  requiredElement("object-name").addEventListener("input", e => updateSelected("object_name", e.target.value));
  requiredElement("distance-steps").addEventListener("input", e => updateSelected("distance_steps", e.target.value ? Number(e.target.value) : null));
  requiredElement("hazard-notes").addEventListener("input", e => updateSelected("notes", e.target.value));
  requiredElement("role-notes").addEventListener("input", e => updateRoleSelected("role_notes", e.target.value));
  requiredElement("image-notes").addEventListener("input", e => { if (state.annotation) { state.annotation.image_notes = e.target.value; state.annotation.status = "draft"; markDirty(); } });
  document.querySelectorAll("[name=category]").forEach(input => input.addEventListener("change", e => updateSelected("hazard_category", e.target.value)));
  document.querySelectorAll("[name=direction]").forEach(input => input.addEventListener("change", e => updateSelected("direction_bin", e.target.value)));
  document.querySelectorAll("[name=evaluation-role]").forEach(input => input.addEventListener("change", e => updateRoleSelected("evaluation_role", e.target.value)));
}

function updateSelected(key, value) {
  const hazard = selectedHazard();
  if (!hazard) return;
  hazard[key] = value;
  state.annotation.status = "draft";
  state.annotation.evaluation_review_status = "pending";
  markDirty();
  renderHazards();
  renderBoxes();
  updateButtons();
}

function updateRoleSelected(key, value) {
  const hazard = selectedHazard();
  if (!hazard) return;
  hazard[key] = value;
  state.annotation.evaluation_review_status = "pending";
  markDirty();
  renderHazards();
  updateButtons();
}

function bindControls() {
  $("status-filter").onchange = e => { state.filter = e.target.value; renderImageList(); };
  $("role-filter").onchange = e => { state.roleFilter = e.target.value; renderImageList(); };
  $("prev-btn").onclick = () => navigate(-1);
  $("next-btn").onclick = () => navigate(1);
  $("next-open-btn").onclick = nextUnfinished;
  $("next-role-pending-btn").onclick = nextRolePending;
  $("select-tool").onclick = () => { state.tool = "select"; updateButtons(); };
  $("draw-tool").onclick = () => { if (!selectedHazard() || selectedHazard().bbox) addHazard(); else { state.tool = "draw"; updateButtons(); } };
  $("add-btn").onclick = () => addHazard();
  $("duplicate-btn").onclick = () => addHazard(selectedHazard());
  $("delete-btn").onclick = deleteSelected;
  $("save-btn").onclick = () => saveAnnotation(true);
  $("complete-btn").onclick = completeAnnotation;
  $("complete-role-review-btn").onclick = completeRoleReview;
  $("backup-btn").onclick = async () => { const result = await request("/api/backup", { method: "POST" }); toast(result.created ? "备份已创建" : "尚无标注文件可备份"); };
  $("zoom-in").onclick = () => changeZoom(1.2);
  $("zoom-out").onclick = () => changeZoom(1 / 1.2);
  $("fit-btn").onclick = fitImage;
  requiredElement("overlay").addEventListener("pointerdown", startDraw);
  requiredElement("overlay").addEventListener("pointermove", moveGesture);
  requiredElement("overlay").addEventListener("pointerup", endGesture);
  requiredElement("overlay").addEventListener("pointercancel", endGesture);

  let pan = null;
  requiredElement("canvas-viewport").addEventListener("pointerdown", e => {
    if (!state.spaceDown) return;
    pan = { x: e.clientX, y: e.clientY, left: e.currentTarget.scrollLeft, top: e.currentTarget.scrollTop };
    e.currentTarget.classList.add("panning");
  });
  window.addEventListener("pointermove", e => {
    if (!pan) return;
    const viewport = $("canvas-viewport");
    viewport.scrollLeft = pan.left - (e.clientX - pan.x);
    viewport.scrollTop = pan.top - (e.clientY - pan.y);
  });
  window.addEventListener("pointerup", () => { pan = null; $("canvas-viewport").classList.remove("panning"); });
  requiredElement("canvas-viewport").addEventListener("wheel", e => { if (e.ctrlKey) { e.preventDefault(); changeZoom(e.deltaY < 0 ? 1.12 : 1 / 1.12); } }, { passive: false });
}

function bindKeys() {
  window.addEventListener("keydown", e => {
    if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    if (e.code === "Space") { state.spaceDown = true; e.preventDefault(); }
    const key = e.key.toLowerCase();
    if (key === "a") navigate(-1);
    if (key === "d") navigate(1);
    if (key === "n") addHazard();
    if (key === "v") { state.tool = "select"; updateButtons(); }
    if (key === "delete") deleteSelected();
    if (e.ctrlKey && key === "s") { e.preventDefault(); saveAnnotation(true); }
    if (key === "enter" && e.ctrlKey) completeAnnotation();
  });
  window.addEventListener("keyup", e => { if (e.code === "Space") state.spaceDown = false; });
  window.addEventListener("beforeunload", e => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
}

async function init() {
  try {
    bindEditor();
    bindControls();
    bindKeys();
    const lastOpened = await refreshImages();
    const initial = state.images.find(img => img.path === lastOpened)?.path || state.images[0]?.path;
    if (initial) await loadImage(initial);
    else { $("empty-canvas").hidden = false; $("image-stage").hidden = true; }
  } catch (error) { toast(error.message, true); }
}

init();
