using System;
using System.Collections.Generic;
using UnityEngine;

namespace ProceduralManholeGenerator
{
    public enum ManholeShape
    {
        Circular,
        Rectangular
    }

    [ExecuteAlways]
    public class ProceduralManhole : MonoBehaviour
    {
        private const string GeneratedPrefix = "PM_";

        [Serializable]
        public class PipeSettings
        {
            public string name = "Pipe";
            [Range(0f, 360f)] public float angleDegrees;
            [Min(0.05f)] public float radius = 0.35f;
            [Min(0.1f)] public float length = 4f;
            [Min(0f)] public float heightFromBottom = 0.9f;
            [Min(0f)] public float intrusionIntoWell = 0.15f;
        }

        [Header("Shape")]
        public ManholeShape shape = ManholeShape.Circular;
        [Min(8)] public int circularSegments = 72;

        [Header("Circular Dimensions")]
        [Min(0.1f)] public float outerRadius = 1.35f;
        [Min(0.1f)] public float openingRadius = 1.1f;

        [Header("Rectangular Dimensions")]
        public Vector2 outerSize = new Vector2(2.7f, 2.7f);
        public Vector2 openingSize = new Vector2(2.1f, 2.1f);

        [Header("Vertical Dimensions")]
        [Min(0.1f)] public float depth = 3.2f;
        [Min(0.02f)] public float floorThickness = 0.16f;

        [Header("Raised Rim")]
        [Min(0.01f)] public float rimHeight = 0.08f;
        [Min(0.02f)] public float rimWidth = 0.18f;
        [Min(0f)] public float rimTopInset = 0.05f;
        [Min(0.02f)] public float coverSeatWidth = 0.14f;
        [Min(0.005f)] public float coverSeatDepth = 0.035f;
        [Min(0f)] public float rimOverhang = 0.03f;
        [Min(0f)] public float bevelHint = 0.015f;

        [Header("Detachable Cover")]
        public bool generateCover = true;
        [Min(0.02f)] public float coverThickness = 0.16f;
        [Min(0f)] public float coverClearance = 0.03f;
        [Min(0f)] public float coverGapAboveRim = 0.02f;
        public bool coverStartsRemoved;
        public bool addCoverGrip = true;

        [Header("Pipes")]
        public List<PipeSettings> pipes = new List<PipeSettings>
        {
            new PipeSettings { name = "Inlet", angleDegrees = 0f },
            new PipeSettings { name = "Outlet", angleDegrees = 180f }
        };

        [Header("Ground Cutout Surface")]
        public bool generateGroundCutoutSurface = true;
        public Vector2 groundSize = new Vector2(6f, 6f);
        [Min(0f)] public float groundOpeningClearance = 0.04f;

        [Header("Output")]
        public bool addColliders = true;
        public bool generatePipeMouths = true;

        [Header("Auto Materials")]
        public bool autoGenerateSimpleMaterials = true;
        public int materialSeed = 12345;
        [Range(64, 1024)] public int generatedTextureSize = 256;
        [Range(0f, 1f)] public float materialVariation = 0.45f;

        [Header("Material Overrides")]
        public Material concreteMaterial;
        public Material pipeMaterial;
        public Material coverMaterial;
        public Material darkMouthMaterial;
        public Material groundMaterial;

        public void Generate()
        {
            SanitizeDimensions();
            ClearGenerated();

            Material concrete = ResolveMaterial(concreteMaterial, new Color(0.36f, 0.36f, 0.34f), "PM_Concrete_Material", ProceduralManholeMaterialKind.Concrete, 0);
            Material pipe = ResolveMaterial(pipeMaterial, new Color(0.22f, 0.22f, 0.22f), "PM_Pipe_Material", ProceduralManholeMaterialKind.Pipe, 19);
            Material cover = ResolveMaterial(coverMaterial, new Color(0.08f, 0.085f, 0.09f), "PM_Cover_Material", ProceduralManholeMaterialKind.CastIronCover, 37);
            Material dark = ResolveMaterial(darkMouthMaterial, new Color(0.02f, 0.018f, 0.016f), "PM_Dark_Material", ProceduralManholeMaterialKind.DarkOpening, 53);
            Material ground = ResolveMaterial(groundMaterial, new Color(0.28f, 0.27f, 0.25f), "PM_Ground_Material", ProceduralManholeMaterialKind.Ground, 71);

            if (shape == ManholeShape.Circular)
            {
                GenerateCircular(concrete, pipe, cover, dark, ground);
            }
            else
            {
                GenerateRectangular(concrete, pipe, cover, dark, ground);
            }
        }

        public void ClearGenerated()
        {
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                Transform child = transform.GetChild(i);
                if (!child.name.StartsWith(GeneratedPrefix, StringComparison.Ordinal))
                {
                    continue;
                }

                if (Application.isPlaying)
                {
                    Destroy(child.gameObject);
                }
                else
                {
                    DestroyImmediate(child.gameObject);
                }
            }
        }

        private void GenerateCircular(Material concrete, Material pipe, Material cover, Material dark, Material ground)
        {
            Transform circularAssembly = CreateGroup("PM_Circular_Assembly");
            float shaftTopY = GetShaftTopY();
            Mesh wallMesh = CreateHollowCylinderMesh(outerRadius, openingRadius, shaftTopY, -depth, circularSegments, false, false);
            CreateMeshObject("PM_Circular_Well_Wall", wallMesh, concrete, addColliders, circularAssembly);

            float coverSeatRadius = GetCircularCoverSeatRadius();
            float rimBottomInnerRadius = Mathf.Max(0.05f, openingRadius - rimOverhang);
            float rimBottomOuterRadius = coverSeatRadius + rimWidth + rimOverhang;
            float rimTopOuterRadius = Mathf.Max(coverSeatRadius + 0.06f, rimBottomOuterRadius - rimTopInset);
            float rimTopInnerRadius = coverSeatRadius + rimTopInset;
            float coverSeatY = GetCoverSeatY();
            Mesh rimMesh = CreateSteppedCircularRimMesh(
                rimBottomOuterRadius,
                rimBottomInnerRadius,
                rimTopOuterRadius,
                rimTopInnerRadius,
                coverSeatRadius,
                openingRadius,
                coverSeatY,
                rimHeight,
                0f,
                circularSegments);
            CreateMeshObject("PM_Raised_Circular_Rim", rimMesh, concrete, addColliders, circularAssembly);

            CreatePrimitive("PM_Circular_Floor", PrimitiveType.Cylinder, concrete,
                new Vector3(openingRadius * 2f, floorThickness * 0.5f, openingRadius * 2f),
                new Vector3(0f, -depth - floorThickness * 0.5f, 0f),
                Quaternion.identity,
                addColliders,
                circularAssembly);

            GeneratePipes(pipe, dark);

            if (generateGroundCutoutSurface)
            {
                float groundHoleRadius = openingRadius + groundOpeningClearance;
                Mesh groundMesh = CreateFlatSquareGroundWithCircularHoleMesh(groundSize, groundHoleRadius, circularSegments);
                CreateMeshObject("PM_Ground_Cutout_Surface", groundMesh, ground, addColliders);
            }

            if (generateCover)
            {
                float coverRadius = Mathf.Max(0.05f, coverSeatRadius - coverClearance);
                GameObject coverObject = CreatePrimitive("PM_Detachable_Cover", PrimitiveType.Cylinder, cover,
                    new Vector3(coverRadius * 2f, coverThickness * 0.5f, coverRadius * 2f),
                    new Vector3(0f, GetCoverRestCenterY(), 0f),
                    Quaternion.identity,
                    addColliders);
                coverObject.SetActive(!coverStartsRemoved);

                if (addCoverGrip)
                {
                    AddCircularCoverGrip(coverObject.transform, cover);
                }
            }
        }

        private void GenerateRectangular(Material concrete, Material pipe, Material cover, Material dark, Material ground)
        {
            float shaftTopY = GetShaftTopY();
            Vector2 halfOuter = outerSize * 0.5f;
            Vector2 halfOpening = openingSize * 0.5f;
            float shaftHeight = Mathf.Max(0.05f, depth + shaftTopY);
            float centerY = (shaftTopY - depth) * 0.5f;

            CreateRectWallPieces("PM_Rect_Well", halfOuter, halfOpening, shaftHeight, centerY, concrete);

            Vector2 rimHalfOuter = halfOpening + Vector2.one * (rimWidth + rimOverhang);
            Vector2 rimHalfInner = new Vector2(
                Mathf.Max(0.05f, halfOpening.x - rimOverhang - bevelHint),
                Mathf.Max(0.05f, halfOpening.y - rimOverhang - bevelHint));
            CreateRectWallPieces("PM_Raised_Rect_Rim", rimHalfOuter, rimHalfInner, rimHeight, rimHeight * 0.5f, concrete);

            CreatePrimitive("PM_Rect_Floor", PrimitiveType.Cube, concrete,
                new Vector3(openingSize.x, floorThickness, openingSize.y),
                new Vector3(0f, -depth - floorThickness * 0.5f, 0f),
                Quaternion.identity,
                addColliders);

            GeneratePipes(pipe, dark);

            if (generateGroundCutoutSurface)
            {
                Vector2 holeHalfSize = halfOpening + Vector2.one * groundOpeningClearance;
                Mesh groundMesh = CreateFlatRectGroundWithRectHoleMesh(groundSize * 0.5f, holeHalfSize);
                CreateMeshObject("PM_Ground_Cutout_Surface", groundMesh, ground, addColliders);
            }

            if (generateCover)
            {
                Vector2 coverSize = new Vector2(
                    Mathf.Max(0.05f, openingSize.x + coverSeatWidth * 2f - coverClearance * 2f),
                    Mathf.Max(0.05f, openingSize.y + coverSeatWidth * 2f - coverClearance * 2f));
                GameObject coverObject = CreatePrimitive("PM_Detachable_Cover", PrimitiveType.Cube, cover,
                    new Vector3(coverSize.x, coverThickness, coverSize.y),
                    new Vector3(0f, GetCoverRestCenterY(), 0f),
                    Quaternion.identity,
                    addColliders);
                coverObject.SetActive(!coverStartsRemoved);

                if (addCoverGrip)
                {
                    AddRectCoverGrip(coverObject.transform, cover, coverSize);
                }
            }
        }

        private void GeneratePipes(Material pipeMaterialResolved, Material darkMaterialResolved)
        {
            foreach (PipeSettings pipeSettings in pipes)
            {
                float radians = pipeSettings.angleDegrees * Mathf.Deg2Rad;
                Vector3 direction = new Vector3(Mathf.Cos(radians), 0f, Mathf.Sin(radians)).normalized;
                float boundary = GetOpeningBoundaryDistance(direction);
                float y = -depth + Mathf.Clamp(pipeSettings.heightFromBottom, 0f, depth);
                Vector3 center = direction * (boundary + pipeSettings.length * 0.5f - pipeSettings.intrusionIntoWell);
                center.y = y;

                Quaternion rotation = Quaternion.FromToRotation(Vector3.up, direction);
                GameObject pipe = CreatePrimitive("PM_" + CleanName(pipeSettings.name), PrimitiveType.Cylinder, pipeMaterialResolved,
                    new Vector3(pipeSettings.radius * 2f, pipeSettings.length * 0.5f, pipeSettings.radius * 2f),
                    center,
                    rotation,
                    addColliders);

                if (!generatePipeMouths)
                {
                    continue;
                }

                GameObject mouth = CreatePrimitive(pipe.name + "_Mouth", PrimitiveType.Cylinder, darkMaterialResolved,
                    new Vector3(pipeSettings.radius * 2.08f, 0.012f, pipeSettings.radius * 2.08f),
                    direction * (boundary - 0.015f) + Vector3.up * y,
                    rotation,
                    false);
                mouth.transform.SetParent(transform, false);
            }
        }

        private void CreateRectWallPieces(string prefix, Vector2 halfOuter, Vector2 halfInner, float height, float centerY, Material material)
        {
            float northSouthDepth = Mathf.Max(0.01f, halfOuter.y - halfInner.y);
            float eastWestWidth = Mathf.Max(0.01f, halfOuter.x - halfInner.x);

            CreatePrimitive(prefix + "_North", PrimitiveType.Cube, material,
                new Vector3(halfOuter.x * 2f, height, northSouthDepth),
                new Vector3(0f, centerY, halfInner.y + northSouthDepth * 0.5f),
                Quaternion.identity,
                addColliders);

            CreatePrimitive(prefix + "_South", PrimitiveType.Cube, material,
                new Vector3(halfOuter.x * 2f, height, northSouthDepth),
                new Vector3(0f, centerY, -halfInner.y - northSouthDepth * 0.5f),
                Quaternion.identity,
                addColliders);

            CreatePrimitive(prefix + "_East", PrimitiveType.Cube, material,
                new Vector3(eastWestWidth, height, halfInner.y * 2f),
                new Vector3(halfInner.x + eastWestWidth * 0.5f, centerY, 0f),
                Quaternion.identity,
                addColliders);

            CreatePrimitive(prefix + "_West", PrimitiveType.Cube, material,
                new Vector3(eastWestWidth, height, halfInner.y * 2f),
                new Vector3(-halfInner.x - eastWestWidth * 0.5f, centerY, 0f),
                Quaternion.identity,
                addColliders);
        }

        private void AddCircularCoverGrip(Transform coverRoot, Material material)
        {
            float y = coverThickness * 0.5f + 0.035f;
            CreatePrimitive("PM_Cover_Grip_A", PrimitiveType.Cube, material,
                new Vector3(openingRadius * 0.75f, 0.06f, 0.09f),
                coverRoot.localPosition + new Vector3(0f, y, 0f),
                Quaternion.identity,
                addColliders).transform.SetParent(coverRoot, true);

            CreatePrimitive("PM_Cover_Grip_B", PrimitiveType.Cube, material,
                new Vector3(0.09f, 0.06f, openingRadius * 0.75f),
                coverRoot.localPosition + new Vector3(0f, y + 0.005f, 0f),
                Quaternion.identity,
                addColliders).transform.SetParent(coverRoot, true);
        }

        private void AddRectCoverGrip(Transform coverRoot, Material material, Vector2 coverSize)
        {
            CreatePrimitive("PM_Cover_Grip", PrimitiveType.Cube, material,
                new Vector3(coverSize.x * 0.55f, 0.06f, 0.1f),
                coverRoot.localPosition + new Vector3(0f, coverThickness * 0.5f + 0.035f, 0f),
                Quaternion.identity,
                addColliders).transform.SetParent(coverRoot, true);
        }

        private Mesh CreateHollowCylinderMesh(float outer, float inner, float topY, float bottomY, int segments, bool capTop, bool capBottom)
        {
            int vertexCount = (segments + 1) * 4;
            List<Vector3> vertices = new List<Vector3>(vertexCount);
            List<Vector2> uvs = new List<Vector2>(vertexCount);
            List<int> triangles = new List<int>(segments * 24);
            float verticalSpan = Mathf.Max(0.001f, Mathf.Abs(topY - bottomY));

            for (int i = 0; i <= segments; i++)
            {
                float a = Mathf.PI * 2f * i / segments;
                float c = Mathf.Cos(a);
                float s = Mathf.Sin(a);
                float u = (float)i / segments;
                vertices.Add(new Vector3(c * outer, topY, s * outer));
                vertices.Add(new Vector3(c * outer, bottomY, s * outer));
                vertices.Add(new Vector3(c * inner, topY, s * inner));
                vertices.Add(new Vector3(c * inner, bottomY, s * inner));
                uvs.Add(new Vector2(u, topY / verticalSpan));
                uvs.Add(new Vector2(u, bottomY / verticalSpan));
                uvs.Add(new Vector2(u, topY / verticalSpan));
                uvs.Add(new Vector2(u, bottomY / verticalSpan));
            }

            for (int i = 0; i < segments; i++)
            {
                int next = i + 1;
                int v = i * 4;
                int n = next * 4;

                AddQuad(triangles, v, n, n + 1, v + 1);
                AddQuad(triangles, v + 2, v + 3, n + 3, n + 2);

                if (capTop)
                {
                    AddQuad(triangles, v, v + 2, n + 2, n);
                }

                if (capBottom)
                {
                    AddQuad(triangles, v + 1, n + 1, n + 3, v + 3);
                }
            }

            Mesh mesh = new Mesh { name = "Procedural Hollow Cylinder" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.SetUVs(0, uvs);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        public float GetCircularCoverSeatRadius()
        {
            return openingRadius + coverSeatWidth;
        }

        public float GetCoverSeatY()
        {
            return Mathf.Max(0.005f, rimHeight - coverSeatDepth);
        }

        public float GetCoverRestCenterY()
        {
            return GetCoverSeatY() + coverGapAboveRim + coverThickness * 0.5f;
        }

        private Mesh CreateSteppedCircularRimMesh(float bottomOuter, float bottomInner, float topOuter, float topInner, float seatOuter, float seatInner, float seatY, float topY, float bottomY, int segments)
        {
            List<Vector3> vertices = new List<Vector3>(segments * 16);
            List<int> triangles = new List<int>(segments * 36);

            for (int i = 0; i < segments; i++)
            {
                int next = (i + 1) % segments;
                float a0 = Mathf.PI * 2f * i / segments;
                float a1 = Mathf.PI * 2f * next / segments;
                Vector3 bo0 = RingPoint(a0, bottomOuter, bottomY);
                Vector3 bo1 = RingPoint(a1, bottomOuter, bottomY);
                Vector3 bi0 = RingPoint(a0, bottomInner, bottomY);
                Vector3 bi1 = RingPoint(a1, bottomInner, bottomY);
                Vector3 to0 = RingPoint(a0, topOuter, topY);
                Vector3 to1 = RingPoint(a1, topOuter, topY);
                Vector3 ti0 = RingPoint(a0, topInner, topY);
                Vector3 ti1 = RingPoint(a1, topInner, topY);
                Vector3 so0 = RingPoint(a0, seatOuter, seatY);
                Vector3 so1 = RingPoint(a1, seatOuter, seatY);
                Vector3 si0 = RingPoint(a0, seatInner, seatY);
                Vector3 si1 = RingPoint(a1, seatInner, seatY);

                AddFace(vertices, triangles, bo0, to0, to1, bo1);
                AddFace(vertices, triangles, to0, ti0, ti1, to1);
                AddFace(vertices, triangles, ti0, so0, so1, ti1);
                AddFace(vertices, triangles, so0, si0, si1, so1);
                AddFace(vertices, triangles, si0, bi0, bi1, si1);
                AddFace(vertices, triangles, bi0, bo0, bo1, bi1);
            }

            Mesh mesh = new Mesh { name = "Procedural Stepped Circular Rim" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.SetUVs(0, CreatePlanarUvsXZ(vertices));
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static List<Vector2> CreatePlanarUvsXZ(List<Vector3> vertices)
        {
            List<Vector2> uvs = new List<Vector2>(vertices.Count);
            for (int i = 0; i < vertices.Count; i++)
            {
                Vector3 vertex = vertices[i];
                uvs.Add(new Vector2(vertex.x, vertex.z));
            }

            return uvs;
        }

        private static List<Vector2> CreateNormalizedPlanarUvsXZ(List<Vector3> vertices, float minX, float maxX, float minZ, float maxZ)
        {
            float width = Mathf.Max(0.001f, maxX - minX);
            float depth = Mathf.Max(0.001f, maxZ - minZ);
            List<Vector2> uvs = new List<Vector2>(vertices.Count);
            for (int i = 0; i < vertices.Count; i++)
            {
                Vector3 vertex = vertices[i];
                uvs.Add(new Vector2((vertex.x - minX) / width, (vertex.z - minZ) / depth));
            }

            return uvs;
        }

        private static Vector3 RingPoint(float angle, float radius, float y)
        {
            return new Vector3(Mathf.Cos(angle) * radius, y, Mathf.Sin(angle) * radius);
        }

        private static void AddFace(List<Vector3> vertices, List<int> triangles, Vector3 a, Vector3 b, Vector3 c, Vector3 d)
        {
            int start = vertices.Count;
            vertices.Add(a);
            vertices.Add(b);
            vertices.Add(c);
            vertices.Add(d);
            AddQuad(triangles, start, start + 1, start + 2, start + 3);
        }

        private Mesh CreateFlatSquareGroundWithCircularHoleMesh(Vector2 size, float holeRadius, int segments)
        {
            Vector2 half = size * 0.5f;
            holeRadius = Mathf.Max(0.05f, Mathf.Min(holeRadius, Mathf.Min(half.x, half.y) - 0.05f));

            List<Vector3> vertices = new List<Vector3>(segments * 2);
            List<int> triangles = new List<int>(segments * 6);

            for (int i = 0; i < segments; i++)
            {
                float a = Mathf.PI * 2f * i / segments;
                float c = Mathf.Cos(a);
                float s = Mathf.Sin(a);
                float edgeDistance = Mathf.Min(
                    Mathf.Abs(c) > 0.0001f ? half.x / Mathf.Abs(c) : float.PositiveInfinity,
                    Mathf.Abs(s) > 0.0001f ? half.y / Mathf.Abs(s) : float.PositiveInfinity);

                vertices.Add(new Vector3(c * edgeDistance, 0f, s * edgeDistance));
                vertices.Add(new Vector3(c * holeRadius, 0f, s * holeRadius));
            }

            for (int i = 0; i < segments; i++)
            {
                int next = (i + 1) % segments;
                int v = i * 2;
                int n = next * 2;
                AddQuad(triangles, v, v + 1, n + 1, n);
            }

            Mesh mesh = new Mesh { name = "Procedural Flat Ground Cutout Surface" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.SetUVs(0, CreateNormalizedPlanarUvsXZ(vertices, -half.x, half.x, -half.y, half.y));
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private Mesh CreateFlatRectGroundWithRectHoleMesh(Vector2 halfOuter, Vector2 halfInner)
        {
            halfInner = new Vector2(
                Mathf.Min(Mathf.Max(0.05f, halfInner.x), halfOuter.x - 0.05f),
                Mathf.Min(Mathf.Max(0.05f, halfInner.y), halfOuter.y - 0.05f));

            List<Vector3> vertices = new List<Vector3>
            {
                new Vector3(-halfOuter.x, 0f, halfOuter.y),
                new Vector3(halfOuter.x, 0f, halfOuter.y),
                new Vector3(halfOuter.x, 0f, halfInner.y),
                new Vector3(-halfOuter.x, 0f, halfInner.y),
                new Vector3(-halfOuter.x, 0f, -halfInner.y),
                new Vector3(halfOuter.x, 0f, -halfInner.y),
                new Vector3(halfOuter.x, 0f, -halfOuter.y),
                new Vector3(-halfOuter.x, 0f, -halfOuter.y),
                new Vector3(-halfOuter.x, 0f, halfInner.y),
                new Vector3(-halfInner.x, 0f, halfInner.y),
                new Vector3(-halfInner.x, 0f, -halfInner.y),
                new Vector3(-halfOuter.x, 0f, -halfInner.y),
                new Vector3(halfInner.x, 0f, halfInner.y),
                new Vector3(halfOuter.x, 0f, halfInner.y),
                new Vector3(halfOuter.x, 0f, -halfInner.y),
                new Vector3(halfInner.x, 0f, -halfInner.y)
            };

            List<int> triangles = new List<int>(24);
            AddQuad(triangles, 0, 1, 2, 3);
            AddQuad(triangles, 4, 5, 6, 7);
            AddQuad(triangles, 8, 9, 10, 11);
            AddQuad(triangles, 12, 13, 14, 15);

            Mesh mesh = new Mesh { name = "Procedural Flat Rect Ground Cutout Surface" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.SetUVs(0, CreateNormalizedPlanarUvsXZ(vertices, -halfOuter.x, halfOuter.x, -halfOuter.y, halfOuter.y));
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private float GetShaftTopY()
        {
            return 0f;
        }

        private Transform CreateGroup(string objectName)
        {
            GameObject group = new GameObject(objectName);
            group.transform.SetParent(transform, false);
            return group.transform;
        }

        private GameObject CreateMeshObject(string objectName, Mesh mesh, Material material, bool withCollider, Transform parent = null)
        {
            GameObject go = new GameObject(objectName);
            go.transform.SetParent(parent != null ? parent : transform, false);
            MeshFilter meshFilter = go.AddComponent<MeshFilter>();
            MeshRenderer meshRenderer = go.AddComponent<MeshRenderer>();
            meshFilter.sharedMesh = mesh;
            meshRenderer.sharedMaterial = material;

            if (withCollider)
            {
                MeshCollider meshCollider = go.AddComponent<MeshCollider>();
                meshCollider.sharedMesh = mesh;
            }

            return go;
        }

        private GameObject CreatePrimitive(string objectName, PrimitiveType type, Material material, Vector3 scale, Vector3 localPosition, Quaternion localRotation, bool withCollider, Transform parent = null)
        {
            GameObject go = GameObject.CreatePrimitive(type);
            go.name = objectName;
            go.transform.SetParent(parent != null ? parent : transform, false);
            go.transform.localPosition = localPosition;
            go.transform.localRotation = localRotation;
            go.transform.localScale = scale;
            Renderer renderer = go.GetComponent<Renderer>();
            renderer.sharedMaterial = material;

            if (!withCollider)
            {
                Collider collider = go.GetComponent<Collider>();
                if (collider != null)
                {
                    if (Application.isPlaying)
                    {
                        Destroy(collider);
                    }
                    else
                    {
                        DestroyImmediate(collider);
                    }
                }
            }

            return go;
        }

        private float GetOpeningBoundaryDistance(Vector3 direction)
        {
            if (shape == ManholeShape.Circular)
            {
                return openingRadius;
            }

            float x = Mathf.Abs(direction.x);
            float z = Mathf.Abs(direction.z);
            Vector2 half = openingSize * 0.5f;

            if (x < 0.0001f)
            {
                return half.y / Mathf.Max(z, 0.0001f);
            }

            if (z < 0.0001f)
            {
                return half.x / Mathf.Max(x, 0.0001f);
            }

            return Mathf.Min(half.x / x, half.y / z);
        }

        private void SanitizeDimensions()
        {
            circularSegments = Mathf.Max(8, circularSegments);
            openingRadius = Mathf.Min(openingRadius, outerRadius - 0.05f);
            openingRadius = Mathf.Max(0.05f, openingRadius);
            rimWidth = Mathf.Clamp(rimWidth, 0.02f, Mathf.Max(0.02f, outerRadius - openingRadius + 0.35f));
            rimTopInset = Mathf.Clamp(rimTopInset, 0f, rimWidth * 0.45f);
            rimHeight = Mathf.Max(0.01f, rimHeight);
            coverSeatWidth = Mathf.Max(0.02f, coverSeatWidth);
            coverSeatDepth = Mathf.Clamp(coverSeatDepth, 0.005f, rimHeight * 0.85f);
            rimOverhang = Mathf.Max(0f, rimOverhang);
            groundSize = new Vector2(Mathf.Max(1f, groundSize.x), Mathf.Max(1f, groundSize.y));
            groundOpeningClearance = Mathf.Max(0f, groundOpeningClearance);
            outerSize = new Vector2(Mathf.Max(0.2f, outerSize.x), Mathf.Max(0.2f, outerSize.y));
            openingSize = new Vector2(
                Mathf.Clamp(openingSize.x, 0.1f, outerSize.x - 0.1f),
                Mathf.Clamp(openingSize.y, 0.1f, outerSize.y - 0.1f));
        }

        private static void AddQuad(List<int> triangles, int a, int b, int c, int d)
        {
            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
            triangles.Add(a);
            triangles.Add(c);
            triangles.Add(d);
        }

        private static string CleanName(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
            {
                return "Pipe";
            }

            foreach (char invalid in System.IO.Path.GetInvalidFileNameChars())
            {
                raw = raw.Replace(invalid, '_');
            }

            return raw.Replace(' ', '_');
        }

        private Material ResolveMaterial(Material source, Color color, string name, ProceduralManholeMaterialKind kind, int seedOffset)
        {
            if (source != null)
            {
                return source;
            }

            if (autoGenerateSimpleMaterials)
            {
                return ProceduralManholeMaterials.Create(kind, materialSeed + seedOffset, generatedTextureSize, materialVariation);
            }

            Shader shader = FindSupportedShader();
            Material material = new Material(shader) { name = name };

            if (material.HasProperty("_BaseColor"))
            {
                material.SetColor("_BaseColor", color);
            }

            if (material.HasProperty("_Color"))
            {
                material.SetColor("_Color", color);
            }

            return material;
        }

        private static Shader FindSupportedShader()
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader != null)
            {
                return shader;
            }

            shader = Shader.Find("HDRP/Lit");
            if (shader != null)
            {
                return shader;
            }

            shader = Shader.Find("Standard");
            if (shader != null)
            {
                return shader;
            }

            return Shader.Find("Sprites/Default");
        }
    }
}
