// PS：该代码抓取的第一张图片是不输出的
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;


[RequireComponent(typeof(Camera))]
public class CameraCaptureAndGT : MonoBehaviour
{
    [Header("Capture Settings")]
    public Camera captureCamera;                   // 不填则自动取本机 Camera
    public string folderPath = "Captures";         // 输出目录（相对路径将基于 Application.dataPath 上一级）
    public int captureWidth = 1920;                // 导出图片宽
    public int captureHeight = 1080;               // 导出图片高
    public float captureInterval = 2f;             // 抓取间隔（秒）
    public enum CaptureMode
    {
        Interval, // 定时抓取
        KeyPress  // 按键抓取
    }

    [Header("Capture Mode")]
    public CaptureMode captureMode = CaptureMode.Interval; // 默认定时
    public KeyCode captureKey = KeyCode.Return; // 按键模式下的触发键（默认 Enter）

    [Header("Filtering Settings")]
    [Tooltip("只遍历这些根目录下的第一层子物体（不包含第二层，如 LOD 等）")]
    public Transform[] rootFolders;                // 例如指向 “Hazard_2D”、“Hazard_3D”
    [Tooltip("水平扇形半角（度）。90° 即 9:00–3:00")]
    public float sectorHalfAngleDeg = 90f;         // 可调
    public float maxDistanceMeters = 20f;          // 距离阈值（米）

    [Header("Misc")]
    public bool includeInactive = false;           // 是否遍历未激活对象
    public bool logToConsole = true;               // 控制台打印简要信息

    // 运行时
    private float _timer;
    private RenderTexture _rt;
    private Texture2D _tex;

    // ==== 数据结构（用于 JSON 序列化） ====

    [Serializable]
    public class CameraParameters
    {
        public float fieldOfView;
        public float aspect;
        public float nearClipPlane;
        public float farClipPlane;
        public float[] position;        // [x,y,z]
        public float[] rotationEuler;   // [x,y,z]
        public float[] forward;         // [x,y,z]
        public float[] up;              // [x,y,z]
        public float[] right;           // [x,y,z]
        public int imageWidth;
        public int imageHeight;
    }

    [Serializable]
    public class ModelInfo
    {
        public string name;
        public float[] model_position;          // 物体 transform.position
        public float horizontal_distance;       // 与摄像机最近距离（米）——对整个模型
        public float[] image_position;          // 最近点在图片中的像素坐标 [x,y]（原点左下）
        public string category; // 模型父类名称，设置为分类
        public float direction; // 模型相较摄像头正前方的角度
    }

    [Serializable]
    public class CaptureRecord
    {
        public string imageName;
        public float[] cameraPosition;          // 仅为冗余，等于 cameraParameters.position
        public CameraParameters cameraParameters;
        public List<ModelInfo> models = new List<ModelInfo>();
    }

    private void Awake()
    {
        if (captureCamera == null) captureCamera = GetComponent<Camera>();
        captureCamera.aspect = (float)captureWidth / captureHeight; // 固定16:9
        // 处理输出目录（相对路径 => 项目根目录（Assets 上一级））
        if (!Path.IsPathRooted(folderPath))
        {
            var projectRoot = Directory.GetParent(Application.dataPath)!.FullName;
            folderPath = Path.Combine(projectRoot, folderPath);
        }
        if (!Directory.Exists(folderPath)) Directory.CreateDirectory(folderPath);

        // 准备渲染资源
        _rt = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        _tex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGBA32, false);
    }

    private void OnDestroy()
    {
        if (_rt != null) _rt.Release();
        if (_tex != null) Destroy(_tex);
    }

    private void Update()
    {
        if (captureMode == CaptureMode.Interval)
        {
            _timer += Time.deltaTime;
            if (_timer >= captureInterval)
            {
                _timer = 0f;
                StartCoroutine(CaptureAtEndOfFrame());
            }
        }
        else if (captureMode == CaptureMode.KeyPress)
        {
            if (Input.GetKeyDown(captureKey))
            {
                StartCoroutine(CaptureAtEndOfFrame());
            }
        }
    }

    private System.Collections.IEnumerator CaptureAtEndOfFrame()
    {
        // 帧渲染结束
        yield return new WaitForEndOfFrame();

        // 第一次预热：只做一次，不保存
        if (_firstFrame)
        {
            _firstFrame = false;
            captureCamera.targetTexture = _rt;
            captureCamera.Render();
            captureCamera.targetTexture = null;
            yield break;
        }

        CaptureOnce();
    }

    private bool _firstFrame = true;

    private void CaptureOnce()
    {   
    // —— 抓图前：备份 & 临时调整（保持画风一致，只去掉会导致单帧模糊的效果）——
        HDAdditionalCameraData hd = captureCamera.GetComponent<HDAdditionalCameraData>();
        // 备份
        HDAdditionalCameraData.AntialiasingMode aaWas = default;
        bool customWas = false, dynResWas = false;
        LayerMask volMaskWas = 0;
        FrameSettings fsBackup = default;
        bool hasHD = (hd != null);

        if (hasHD)
        {
            aaWas      = hd.antialiasing;
            customWas  = hd.customRenderingSettings;
            dynResWas  = hd.allowDynamicResolution;
            volMaskWas = hd.volumeLayerMask;
            fsBackup   = hd.renderingPathCustomFrameSettings;

            hd.customRenderingSettings = true;

            // 拿到一份可改的副本（保留 Postprocess：true）
            var fs = hd.renderingPathCustomFrameSettings;
            fs.SetEnabled(FrameSettingsField.Postprocess,    true);   // 关键：保留后期（颜色/色调映射不变）
            fs.SetEnabled(FrameSettingsField.DepthOfField,   false);  // 关闭景深（避免单帧糊）
            fs.SetEnabled(FrameSettingsField.MotionBlur,     false);  // 关闭运动模糊
            fs.SetEnabled(FrameSettingsField.MotionVectors,  false);  // 关闭运动向量（TAA用不到了）
            hd.renderingPathCustomFrameSettings = fs;

            // 抗锯齿：单帧用 SMAA（或 None）
            hd.antialiasing = HDAdditionalCameraData.AntialiasingMode.SubpixelMorphologicalAntiAliasing; // 或 None
            // 禁用动态分辨率（避免抓到低分渲染）
            hd.allowDynamicResolution = false;
            // 不改 volumeLayerMask（保持吃到全局后期），这样画风不变
        }


        // 1) 生成时间戳文件名
        string ts = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
        string imgName = $"IMG_{ts}.png";
        string imgPath = Path.Combine(folderPath, imgName);
        string jsonPath = Path.Combine(folderPath, $"IMG_{ts}.json");

        // 2) 抓图（使用指定分辨率）
        var previous = captureCamera.targetTexture;
        captureCamera.targetTexture = _rt;
        captureCamera.Render();
        RenderTexture.active = _rt;
        _tex.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        _tex.Apply();
        captureCamera.targetTexture = previous;
        RenderTexture.active = null;

        byte[] png = _tex.EncodeToPNG();
        File.WriteAllBytes(imgPath, png);

        // 3) 收集摄像机参数
        var cam = captureCamera;
        var camPos = cam.transform.position;
        var camFwd = cam.transform.forward;
        var camRight = cam.transform.right;
        var camUp = cam.transform.up;

        CameraParameters camParams = new CameraParameters
        {
            fieldOfView = cam.fieldOfView,
            aspect = cam.aspect,
            nearClipPlane = cam.nearClipPlane,
            farClipPlane = cam.farClipPlane,
            position = V3(camPos),
            rotationEuler = V3(cam.transform.rotation.eulerAngles),
            forward = V3(camFwd),
            up = V3(camUp),
            right = V3(camRight),
            imageWidth = captureWidth,
            imageHeight = captureHeight
        };

        // 4) 遍历根目录第一层子物体，做区域与距离过滤，计算最近点与像素坐标
        List<ModelInfo> kept = new List<ModelInfo>();
        float half = Mathf.Clamp(sectorHalfAngleDeg, 0f, 179.9f);

        foreach (var root in rootFolders)
        {
            if (root == null) continue;

            foreach (Transform child in root)
            {
                if (!includeInactive && !child.gameObject.activeInHierarchy) continue;

                // 仅第一层：child 即一个“模型”，不再把 child 的子节点当成独立模型
                var go = child.gameObject;
                Vector3 modelPos = go.transform.position;

                Vector3 toObj = modelPos - camPos;

                // 在相机前方？
                bool inFront = Vector3.Dot(camFwd, toObj) > 0f;
                if (!inFront) continue;

                // 水平向量
                Vector3 toObjXZ = Vector3.ProjectOnPlane(toObj, Vector3.up);
                Vector3 camFwdXZ = Vector3.ProjectOnPlane(camFwd, Vector3.up);

                // 若几乎在脚下，认为方向角为 0
                bool nearFoot = toObjXZ.sqrMagnitude < 0.01f;

                // 有符号角：Unity 以向上轴（Y）为正旋转，默认左正右负；取负号后即右正左负
                float directionDeg = 0f;
                if (!nearFoot)
                {
                    directionDeg = -Vector3.SignedAngle(camFwdXZ, toObjXZ, Vector3.up); // 右正、左负
                    // 规范到 [-180, 180]
                    if (directionDeg > 180f) directionDeg -= 360f;
                    if (directionDeg < -180f) directionDeg += 360f;
                }

                // 扇形过滤（仍然使用半角 half），脚下近距离直接放行
                if (!nearFoot && Mathf.Abs(directionDeg) > half) continue;

                // —— 水平扇形判断（把向量投影到水平面）——
                //Vector3 toObj = modelPos - camPos;
                //Vector3 toObjXZ = Vector3.ProjectOnPlane(toObj, Vector3.up);
                //Vector3 camFwdXZ = Vector3.ProjectOnPlane(camFwd, Vector3.up);

                // 不在前半平面：z<=0（相对摄像机前向的投影长度为负）也可以通过角度判断来统一
                //if (toObjXZ.sqrMagnitude < 1e-6f) continue; // 几乎重合，跳过
                //float angle = Vector3.Angle(camFwdXZ, toObjXZ);
                //if (angle > half) continue; // 超出水平扇形

                // —— 距离过滤（3D 距离）——
                float dist = toObj.magnitude;
                if (dist > maxDistanceMeters) continue;

                // —— 计算“对整个模型”的最近点与距离（优先 Collider，其次 Renderer.bounds）——
                bool found = false;
                Vector3 closest = Vector3.zero;
                float bestSqr = float.PositiveInfinity;

                // 1) 尝试所有 Collider（包含子节点，用于涵盖真实几何）
                var colliders = go.GetComponentsInChildren<Collider>(true);
                if (colliders != null && colliders.Length > 0)
                {
                    foreach (var col in colliders)
                    {
                        Vector3 p = col.ClosestPoint(camPos);
                        float d2 = (p - camPos).sqrMagnitude;
                        if (d2 < bestSqr)
                        {
                            bestSqr = d2;
                            closest = p;
                            found = true;
                        }
                    }
                }

                // 2) 若无 Collider，则回退到 Renderer.bounds（近似）
                if (!found)
                {
                    var renderers = go.GetComponentsInChildren<Renderer>(true);
                    foreach (var rend in renderers)
                    {
                        Bounds b = rend.bounds;
                        Vector3 p = b.ClosestPoint(camPos);
                        float d2 = (p - camPos).sqrMagnitude;
                        if (d2 < bestSqr)
                        {
                            bestSqr = d2;
                            closest = p;
                            found = true;
                        }
                    }
                }

                if (!found) continue; // 找不到几何信息则忽略

                float nearestDist = Mathf.Sqrt(bestSqr);
                // —— 用最近点来算方向角 ——
                Vector3 toClosest = closest - camPos;
                Vector3 toClosestXZ = Vector3.ProjectOnPlane(toClosest, Vector3.up);
                float directionDegClosest = 0f;
                if (toClosestXZ.sqrMagnitude >= 0.01f)
                {
                    Vector3 camFwdXZ2 = Vector3.ProjectOnPlane(camFwd, Vector3.up);
                    directionDegClosest = -Vector3.SignedAngle(camFwdXZ2, toClosestXZ, Vector3.up);
                    if (directionDegClosest > 180f) directionDegClosest -= 360f;
                    if (directionDegClosest < -180f) directionDegClosest += 360f;
                }
                // —— 最近点投影到图像坐标（像素）——
                // 使用与截图一致的投影分辨率
                Vector3 sp = WorldToScreenPointWithSize(cam, closest, captureWidth, captureHeight);
                // 仅记录在前方 & 在屏幕内的点（z>0 且 0..w, 0..h）
                if (sp.z <= 0f) continue;

                // 注意：Unity 的屏幕原点在左下角（与保存的 PNG 一致）
                float px = Mathf.Clamp(sp.x, 0, captureWidth - 1);
                float py = Mathf.Clamp(sp.y, 0, captureHeight - 1);

                kept.Add(new ModelInfo
                {
                    name = go.name,
                    model_position = V3(modelPos),
                    horizontal_distance = nearestDist,
                    image_position = new float[] { px, py },
                    category = root.name,
                    direction = directionDegClosest
                });
            }
        }

        // 5) 组装 JSON
        CaptureRecord rec = new CaptureRecord
        {
            imageName = imgName,
            cameraPosition = V3(camPos),
            cameraParameters = camParams,
            models = kept
        };

        if (hasHD)
        {
            hd.renderingPathCustomFrameSettings = fsBackup;
            hd.customRenderingSettings = customWas;
            hd.antialiasing = aaWas;
            hd.volumeLayerMask = volMaskWas;
            hd.allowDynamicResolution = dynResWas;
        }
        // 用 Unity 内置 JsonUtility（字段需 public）
        string json = JsonUtility.ToJson(rec, true);
        File.WriteAllText(jsonPath, json);

        if (logToConsole)
        {
            Debug.Log($"[CameraCaptureAndGT] Saved {imgName} & JSON with {kept.Count} models at: {folderPath}");
        }
    }

    // 工具：把 Vector3 转成 float[3]
    private static float[] V3(Vector3 v) => new float[] { v.x, v.y, v.z };

    // 在指定分辨率下进行投影（与抓图一致），确保像素坐标匹配导出的 PNG
    private static Vector3 WorldToScreenPointWithSize(Camera cam, Vector3 world, int width, int height)
    {
        // 备份
        var prevAspect = cam.aspect;
        var prevTarget = cam.targetTexture;

        // 临时设置 aspect 以吻合输出宽高（投影矩阵受 aspect 影响）
        cam.aspect = (float)width / height;
        // 不需要实际 RenderTexture，只要用投影矩阵计算
        Vector3 sp = cam.WorldToScreenPoint(world);

        // 还原
        cam.aspect = prevAspect;
        cam.targetTexture = prevTarget;

        // 注意：当 Game 视图分辨率与输出分辨率不一致时，
        // WorldToScreenPoint 返回的是当前 Game 视图分辨率下的像素。
        // 为了严谨，把它归一化再映射到目标分辨率：
        // 但是 Unity 无法直接给出当前 Game 视图像素尺寸，这里采用归一化近似：
        // 屏幕坐标 = Viewport * (width,height)
        Vector3 vp = cam.WorldToViewportPoint(world);
        return new Vector3(vp.x * width, vp.y * height, sp.z);
    }
}
