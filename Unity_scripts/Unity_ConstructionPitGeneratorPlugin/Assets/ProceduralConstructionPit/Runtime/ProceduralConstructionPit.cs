using System;
using System.Collections.Generic;
using UnityEngine;

namespace ProceduralConstructionPitGenerator
{
    public enum PitFootprintMode
    {
        Blob,
        Trench
    }

    [ExecuteAlways]
    [DisallowMultipleComponent]
    public class ProceduralConstructionPit : MonoBehaviour
    {
        private const string GeneratedPrefix = "PCP_";

        [Header("Seed")]
        public int generationSeed = 20260513;

        [Header("Ground")]
        public Vector2 groundSize = new Vector2(8f, 8f);
        [Tooltip("Positive values expand the ground opening. Negative values shrink it inward toward the pit.")]
        [Range(-0.2f, 0.2f)] public float groundOpeningClearance = -0.03f;

        [Header("Footprint")]
        public PitFootprintMode footprintMode = PitFootprintMode.Blob;

        [Header("Pit Shape Ranges")]
        [Min(3)] public int minBoundaryPoints = 9;
        [Min(3)] public int maxBoundaryPoints = 16;
        [Min(0.2f)] public float minRadius = 1.2f;
        [Min(0.2f)] public float maxRadius = 2.1f;
        [Min(0.1f)] public float minDepth = 1.2f;
        [Min(0.1f)] public float maxDepth = 3.0f;
        [Range(0f, 1f)] public float edgeIrregularity = 0.45f;
        [Range(0f, 1f)] public float wallCollapseAmount = 0.35f;
        [Range(0f, 1f)] public float wallRoughness = 0.38f;
        [Min(1)] public int wallVerticalSegments = 4;

        [Header("Trench Shape Ranges")]
        [Min(0.5f)] public float minTrenchLength = 3.2f;
        [Min(0.5f)] public float maxTrenchLength = 5.6f;
        [Min(0.2f)] public float minTrenchWidth = 0.8f;
        [Min(0.2f)] public float maxTrenchWidth = 1.5f;
        [Range(0f, 1f)] public float trenchCurveAmount = 0.35f;
        [Min(2)] public int trenchSegments = 7;

        [Header("Debris")]
        public bool generateDebris = true;
        [Range(0f, 1f)] public float debrisAmount = 0.45f;
        [Min(0)] public int maxDebrisPieces = 28;

        [Header("Broken Edge")]
        public bool generateBrokenEdge = true;
        [Range(0f, 1f)] public float brokenEdgeAmount = 0.55f;
        [Min(0)] public int maxBrokenEdgePieces = 36;
        public bool brokenEdgeUsesGroundMaterial = true;

        [Header("Output")]
        public bool addColliders = true;
        public bool autoGenerateSimpleMaterials = true;
        [Range(64, 1024)] public int generatedTextureSize = 256;
        [Range(0f, 1f)] public float materialVariation = 0.45f;

        [Header("Material Overrides")]
        public Material groundMaterial;
        public Material soilMaterial;
        public Material darkPitMaterial;
        public Material debrisMaterial;
        public Material brokenEdgeMaterial;

        public void Generate()
        {
            Sanitize();
            ClearGenerated();

            System.Random random = new System.Random(generationSeed);
            PitShape shape = NormalizePitShapeWinding(footprintMode == PitFootprintMode.Trench ? CreateTrenchShape(random) : CreateBlobShape(random));

            Material ground = ResolveMaterial(groundMaterial, ConstructionPitMaterialKind.Ground, 0);
            Material soil = ResolveMaterial(soilMaterial, ConstructionPitMaterialKind.Soil, 17);
            Material dark = ResolveMaterial(darkPitMaterial, ConstructionPitMaterialKind.DarkPit, 31);
            Material debris = ResolveMaterial(debrisMaterial, ConstructionPitMaterialKind.Debris, 47);
            Material brokenEdge = ResolveBrokenEdgeMaterial(ground);

            Transform groundGroup = CreateGroup("PCP_Ground");
            Transform pitGroup = CreateGroup("PCP_Pit");
            Transform debrisGroup = CreateGroup("PCP_Debris");

            CreateMeshObject("PCP_Ground_Cutout_Surface", CreateGroundMesh(shape), ground, addColliders, groundGroup);
            if (generateBrokenEdge)
            {
                GenerateBrokenEdge(random, shape, brokenEdge, groundGroup);
            }

            CreateMeshObject("PCP_Pit_Walls", CreatePitWallMesh(shape, random), soil, addColliders, pitGroup);
            CreateMeshObject("PCP_Pit_Floor", CreatePitFloorMesh(shape), dark, addColliders, pitGroup);

            if (generateDebris)
            {
                GenerateDebris(random, shape, debris, debrisGroup);
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

        private static PitShape NormalizePitShapeWinding(PitShape shape)
        {
            if (SignedAreaXZ(shape.Top) >= 0f)
            {
                return shape;
            }

            shape.Top.Reverse();
            shape.Bottom.Reverse();
            return shape;
        }

        private PitShape CreateBlobShape(System.Random random)
        {
            int count = random.Next(minBoundaryPoints, maxBoundaryPoints + 1);
            float depth = Range(random, minDepth, maxDepth);
            List<Vector3> top = new List<Vector3>(count);
            List<Vector3> bottom = new List<Vector3>(count);

            float angleOffset = Range(random, 0f, Mathf.PI * 2f);
            for (int i = 0; i < count; i++)
            {
                float t = (float)i / count;
                float angle = angleOffset + Mathf.PI * 2f * t + Range(random, -0.08f, 0.08f);
                float baseRadius = Range(random, minRadius, maxRadius);
                float radiusScale = Mathf.Lerp(1f, Range(random, 0.55f, 1.35f), edgeIrregularity);
                float topRadius = baseRadius * radiusScale;
                float bottomRadius = topRadius * Mathf.Lerp(0.45f, 0.82f, 1f - wallCollapseAmount) * Range(random, 0.85f, 1.12f);

                top.Add(new Vector3(Mathf.Cos(angle) * topRadius, 0f, Mathf.Sin(angle) * topRadius));
                bottom.Add(new Vector3(Mathf.Cos(angle) * bottomRadius, -depth * Range(random, 0.86f, 1.08f), Mathf.Sin(angle) * bottomRadius));
            }

            return new PitShape(top, bottom, depth);
        }

        private PitShape CreateTrenchShape(System.Random random)
        {
            float boundaryDetailScale = Mathf.Max(0.25f, Range(random, minBoundaryPoints, maxBoundaryPoints) / 12f);
            int segments = Mathf.Max(2, Mathf.RoundToInt(trenchSegments * boundaryDetailScale));
            int count = (segments + 1) * 2;
            float sharedSizeScale = Mathf.Max(0.2f, Range(random, minRadius, maxRadius) / 1.65f);
            float length = Range(random, minTrenchLength, maxTrenchLength) * sharedSizeScale;
            float width = Range(random, minTrenchWidth, maxTrenchWidth) * sharedSizeScale;
            float depth = Range(random, minDepth, maxDepth);
            float rotation = Range(random, 0f, Mathf.PI * 2f);
            Vector3 forward = new Vector3(Mathf.Cos(rotation), 0f, Mathf.Sin(rotation));
            Vector3 right = new Vector3(-forward.z, 0f, forward.x);
            float curveAmplitude = length * 0.16f * trenchCurveAmount;

            Vector3[] centerline = new Vector3[segments + 1];
            for (int i = 0; i <= segments; i++)
            {
                float t = (float)i / segments;
                float along = Mathf.Lerp(-length * 0.5f, length * 0.5f, t);
                float curve = Mathf.Sin(t * Mathf.PI * 2f + Range(random, -0.35f, 0.35f)) * curveAmplitude;
                centerline[i] = forward * along + right * curve;
            }

            List<Vector3> left = new List<Vector3>(segments + 1);
            List<Vector3> rightSide = new List<Vector3>(segments + 1);

            for (int i = 0; i <= segments; i++)
            {
                Vector3 tangent;
                if (i == 0)
                {
                    tangent = (centerline[i + 1] - centerline[i]).normalized;
                }
                else if (i == segments)
                {
                    tangent = (centerline[i] - centerline[i - 1]).normalized;
                }
                else
                {
                    tangent = (centerline[i + 1] - centerline[i - 1]).normalized;
                }

                Vector3 normal = new Vector3(-tangent.z, 0f, tangent.x);
                float localWidth = width * Mathf.Lerp(1f, Range(random, 0.7f, 1.25f), edgeIrregularity);
                float jitter = width * 0.12f * edgeIrregularity;
                left.Add(centerline[i] + normal * (localWidth * 0.5f + Range(random, -jitter, jitter)));
                rightSide.Add(centerline[i] - normal * (localWidth * 0.5f + Range(random, -jitter, jitter)));
            }

            List<Vector3> top = new List<Vector3>(count);
            top.AddRange(left);
            for (int i = rightSide.Count - 1; i >= 0; i--)
            {
                top.Add(rightSide[i]);
            }

            List<Vector3> bottom = new List<Vector3>(count);
            float collapseScale = Mathf.Lerp(0.45f, 0.82f, 1f - wallCollapseAmount);
            Vector3 center = Average(top);
            for (int i = 0; i < top.Count; i++)
            {
                Vector3 p = top[i];
                Vector3 towardCenter = center - p;
                towardCenter.y = 0f;
                towardCenter = towardCenter.sqrMagnitude > 0.0001f ? towardCenter.normalized : Vector3.zero;
                bottom.Add(p + towardCenter * width * (1f - collapseScale) * Range(random, 0.65f, 1.15f) + Vector3.down * depth * Range(random, 0.88f, 1.08f));
            }

            return new PitShape(top, bottom, depth);
        }

        private Mesh CreateGroundMesh(PitShape shape)
        {
            Vector2 half = groundSize * 0.5f;
            List<Vector3> hole = CreateGroundHolePolygon(shape);
            List<Vector3> vertices = CreateBridgedGroundPolygon(hole, half);
            List<int> triangles = TriangulatePolygonXZ(vertices);

            return CreateMesh(
                "Procedural Construction Pit Ground",
                vertices,
                triangles,
                CreateNormalizedPlanarUvsXZ(vertices, -half.x, half.x, -half.y, half.y));
        }

        private static List<Vector3> CreateBridgedGroundPolygon(List<Vector3> hole, Vector2 half)
        {
            int bridgeHoleIndex = 0;
            for (int i = 1; i < hole.Count; i++)
            {
                if (hole[i].x > hole[bridgeHoleIndex].x)
                {
                    bridgeHoleIndex = i;
                }
            }

            Vector3 holeBridge = hole[bridgeHoleIndex];
            Vector3 rectBridge = new Vector3(half.x, 0f, Mathf.Clamp(holeBridge.z, -half.y + 0.001f, half.y - 0.001f));
            List<Vector3> polygon = new List<Vector3>(hole.Count + 8)
            {
                rectBridge,
                new Vector3(half.x, 0f, half.y),
                new Vector3(-half.x, 0f, half.y),
                new Vector3(-half.x, 0f, -half.y),
                new Vector3(half.x, 0f, -half.y),
                rectBridge,
                holeBridge
            };

            for (int offset = 1; offset < hole.Count; offset++)
            {
                int index = (bridgeHoleIndex - offset + hole.Count) % hole.Count;
                polygon.Add(hole[index]);
            }

            polygon.Add(holeBridge);
            return polygon;
        }

        private static List<int> TriangulatePolygonXZ(List<Vector3> vertices)
        {
            List<int> triangles = new List<int>();
            List<int> indices = new List<int>(vertices.Count);
            for (int i = 0; i < vertices.Count; i++)
            {
                indices.Add(i);
            }

            bool clockwise = SignedAreaXZ(vertices) < 0f;
            int guard = 0;
            while (indices.Count > 3 && guard < vertices.Count * vertices.Count)
            {
                bool clipped = false;
                for (int i = 0; i < indices.Count; i++)
                {
                    int previous = indices[(i - 1 + indices.Count) % indices.Count];
                    int current = indices[i];
                    int next = indices[(i + 1) % indices.Count];

                    if (!IsEar(vertices, indices, previous, current, next, clockwise))
                    {
                        continue;
                    }

                    AddTriangleWithOrientation(triangles, previous, current, next, clockwise);
                    indices.RemoveAt(i);
                    clipped = true;
                    break;
                }

                if (!clipped)
                {
                    break;
                }

                guard++;
            }

            if (indices.Count == 3)
            {
                AddTriangleWithOrientation(triangles, indices[0], indices[1], indices[2], clockwise);
            }

            return triangles;
        }

        private static bool IsEar(List<Vector3> vertices, List<int> indices, int previous, int current, int next, bool clockwise)
        {
            if (!IsConvexXZ(vertices[previous], vertices[current], vertices[next], clockwise))
            {
                return false;
            }

            for (int i = 0; i < indices.Count; i++)
            {
                int index = indices[i];
                if (index == previous || index == current || index == next)
                {
                    continue;
                }

                if (PointInTriangleXZ(vertices[index], vertices[previous], vertices[current], vertices[next]))
                {
                    return false;
                }
            }

            return true;
        }

        private static bool IsConvexXZ(Vector3 a, Vector3 b, Vector3 c, bool clockwise)
        {
            float cross = CrossXZ(a, b, c);
            return clockwise ? cross < -0.000001f : cross > 0.000001f;
        }

        private static float CrossXZ(Vector3 a, Vector3 b, Vector3 c)
        {
            return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
        }

        private static bool PointInTriangleXZ(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            const float epsilon = 0.000001f;
            float c1 = CrossXZ(p, a, b);
            float c2 = CrossXZ(p, b, c);
            float c3 = CrossXZ(p, c, a);
            bool strictlyPositive = c1 > epsilon && c2 > epsilon && c3 > epsilon;
            bool strictlyNegative = c1 < -epsilon && c2 < -epsilon && c3 < -epsilon;
            return strictlyPositive || strictlyNegative;
        }

        private static float SignedAreaXZ(List<Vector3> vertices)
        {
            float area = 0f;
            for (int i = 0; i < vertices.Count; i++)
            {
                Vector3 a = vertices[i];
                Vector3 b = vertices[(i + 1) % vertices.Count];
                area += a.x * b.z - b.x * a.z;
            }

            return area * 0.5f;
        }

        private static void AddTriangleWithOrientation(List<int> triangles, int a, int b, int c, bool clockwise)
        {
            if (clockwise)
            {
                triangles.Add(a);
                triangles.Add(b);
                triangles.Add(c);
            }
            else
            {
                triangles.Add(a);
                triangles.Add(c);
                triangles.Add(b);
            }
        }

        private List<Vector3> CreateGroundHolePolygon(PitShape shape)
        {
            List<Vector3> hole = new List<Vector3>(shape.Top.Count);
            Vector3 center = Average(shape.Top);
            for (int i = 0; i < shape.Top.Count; i++)
            {
                Vector3 inner = shape.Top[i];
                Vector3 direction = new Vector3(inner.x - center.x, 0f, inner.z - center.z).normalized;
                hole.Add(inner + direction * groundOpeningClearance);
            }

            return hole;
        }

        private Mesh CreatePitWallMesh(PitShape shape, System.Random random)
        {
            int count = shape.Top.Count;
            int verticalSegments = Mathf.Max(1, wallVerticalSegments);
            List<Vector3> vertices = new List<Vector3>(count * (verticalSegments + 1));
            List<int> triangles = new List<int>(count * verticalSegments * 6);

            for (int i = 0; i < count; i++)
            {
                Vector3 top = shape.Top[i];
                Vector3 bottom = shape.Bottom[i];
                Vector3 radial = new Vector3(top.x, 0f, top.z).normalized;

                for (int layer = 0; layer <= verticalSegments; layer++)
                {
                    float t = (float)layer / verticalSegments;
                    Vector3 point = Vector3.Lerp(top, bottom, t);

                    if (layer > 0 && layer < verticalSegments && wallRoughness > 0f)
                    {
                        float depthScale = Mathf.Sin(t * Mathf.PI);
                        float outward = Range(random, -0.18f, 0.18f) * wallRoughness * depthScale;
                        float vertical = Range(random, -0.08f, 0.08f) * wallRoughness * depthScale;
                        point += radial * outward + Vector3.up * vertical;
                    }

                    vertices.Add(point);
                }
            }

            for (int i = 0; i < count; i++)
            {
                int next = (i + 1) % count;

                for (int layer = 0; layer < verticalSegments; layer++)
                {
                    int a = i * (verticalSegments + 1) + layer;
                    int b = a + 1;
                    int c = next * (verticalSegments + 1) + layer + 1;
                    int d = next * (verticalSegments + 1) + layer;
                    AddQuad(triangles, a, b, c, d);
                }
            }

            return CreateMesh("Procedural Construction Pit Walls", vertices, triangles);
        }

        private Mesh CreatePitFloorMesh(PitShape shape)
        {
            List<Vector3> vertices = new List<Vector3> { Average(shape.Bottom) };
            vertices.AddRange(shape.Bottom);
            List<int> triangles = new List<int>(shape.Bottom.Count * 3);

            for (int i = 0; i < shape.Bottom.Count; i++)
            {
                int next = (i + 1) % shape.Bottom.Count;
                triangles.Add(0);
                triangles.Add(next + 1);
                triangles.Add(i + 1);
            }

            return CreateMesh("Procedural Construction Pit Floor", vertices, triangles);
        }

        private void GenerateDebris(System.Random random, PitShape shape, Material material, Transform parent)
        {
            int count = Mathf.RoundToInt(maxDebrisPieces * debrisAmount);
            for (int i = 0; i < count; i++)
            {
                Vector3 edge = shape.Top[random.Next(0, shape.Top.Count)];
                Vector3 outward = new Vector3(edge.x, 0f, edge.z).normalized;
                Vector3 position = edge + outward * Range(random, 0.15f, 0.75f);
                position += new Vector3(Range(random, -0.18f, 0.18f), 0.03f, Range(random, -0.18f, 0.18f));

                PrimitiveType primitiveType = random.NextDouble() > 0.35 ? PrimitiveType.Cube : PrimitiveType.Sphere;
                GameObject debris = GameObject.CreatePrimitive(primitiveType);
                debris.name = GeneratedPrefix + "Debris_" + i.ToString("00");
                debris.transform.SetParent(parent, false);
                debris.transform.localPosition = position;
                debris.transform.localRotation = Quaternion.Euler(Range(random, 0f, 360f), Range(random, 0f, 360f), Range(random, 0f, 360f));
                float scale = Range(random, 0.08f, 0.26f);
                debris.transform.localScale = new Vector3(scale * Range(random, 0.7f, 1.6f), scale * Range(random, 0.35f, 0.9f), scale * Range(random, 0.7f, 1.5f));
                debris.GetComponent<Renderer>().sharedMaterial = material;

                if (!addColliders)
                {
                    Collider collider = debris.GetComponent<Collider>();
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
            }
        }

        private void GenerateBrokenEdge(System.Random random, PitShape shape, Material material, Transform parent)
        {
            int count = Mathf.RoundToInt(maxBrokenEdgePieces * brokenEdgeAmount);
            for (int i = 0; i < count; i++)
            {
                int index = random.Next(0, shape.Top.Count);
                Vector3 a = shape.Top[index];
                Vector3 b = shape.Top[(index + 1) % shape.Top.Count];
                Vector3 edgeCenter = Vector3.Lerp(a, b, Range(random, 0.15f, 0.85f));
                Vector3 outward = new Vector3(edgeCenter.x, 0f, edgeCenter.z).normalized;
                if (outward.sqrMagnitude < 0.0001f)
                {
                    outward = new Vector3(1f, 0f, 0f);
                }

                Vector3 tangent = (b - a);
                tangent.y = 0f;
                tangent = tangent.sqrMagnitude > 0.0001f ? tangent.normalized : new Vector3(-outward.z, 0f, outward.x);
                Vector3 position = edgeCenter + outward * Range(random, 0.03f, 0.24f) + tangent * Range(random, -0.08f, 0.08f);
                position.y = 0.012f;

                GameObject piece = GameObject.CreatePrimitive(PrimitiveType.Cube);
                piece.name = GeneratedPrefix + "BrokenEdge_" + i.ToString("00");
                piece.transform.SetParent(parent, false);
                piece.transform.localPosition = position;
                piece.transform.localRotation = Quaternion.LookRotation(tangent, Vector3.up) * Quaternion.Euler(0f, Range(random, -28f, 28f), 0f);
                piece.transform.localScale = new Vector3(Range(random, 0.18f, 0.46f), Range(random, 0.012f, 0.035f), Range(random, 0.08f, 0.24f));
                piece.GetComponent<Renderer>().sharedMaterial = material;

                if (!addColliders)
                {
                    Collider collider = piece.GetComponent<Collider>();
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
            }
        }

        private Transform CreateGroup(string groupName)
        {
            GameObject group = new GameObject(groupName);
            group.transform.SetParent(transform, false);
            return group.transform;
        }

        private GameObject CreateMeshObject(string objectName, Mesh mesh, Material material, bool withCollider, Transform parent)
        {
            GameObject go = new GameObject(objectName);
            go.transform.SetParent(parent, false);
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

        private Material ResolveBrokenEdgeMaterial(Material ground)
        {
            if (brokenEdgeUsesGroundMaterial || brokenEdgeMaterial == null)
            {
                return ground;
            }

            return brokenEdgeMaterial;
        }

        private Material ResolveMaterial(Material source, ConstructionPitMaterialKind kind, int seedOffset)
        {
            if (source != null)
            {
                return source;
            }

            if (autoGenerateSimpleMaterials)
            {
                return ConstructionPitMaterials.Create(kind, generationSeed + seedOffset, generatedTextureSize, materialVariation);
            }

            return new Material(ConstructionPitMaterials.FindSupportedShader()) { color = Color.gray };
        }

        private void Sanitize()
        {
            groundSize = new Vector2(Mathf.Max(1f, groundSize.x), Mathf.Max(1f, groundSize.y));
            maxBoundaryPoints = Mathf.Max(minBoundaryPoints, maxBoundaryPoints);
            maxRadius = Mathf.Max(minRadius, maxRadius);
            maxDepth = Mathf.Max(minDepth, maxDepth);
            maxTrenchLength = Mathf.Max(minTrenchLength, maxTrenchLength);
            maxTrenchWidth = Mathf.Max(minTrenchWidth, maxTrenchWidth);
            trenchSegments = Mathf.Max(2, trenchSegments);
            wallVerticalSegments = Mathf.Max(1, wallVerticalSegments);
            groundOpeningClearance = Mathf.Clamp(groundOpeningClearance, -0.2f, 0.2f);
            maxDebrisPieces = Mathf.Max(0, maxDebrisPieces);
            maxBrokenEdgePieces = Mathf.Max(0, maxBrokenEdgePieces);
        }

        private static Mesh CreateMesh(string name, List<Vector3> vertices, List<int> triangles, List<Vector2> uvs = null)
        {
            Mesh mesh = new Mesh { name = name };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.SetUVs(0, uvs ?? CreatePlanarUvsXZ(vertices));
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

        private static void AddFace(List<Vector3> vertices, List<int> triangles, Vector3 a, Vector3 b, Vector3 c, Vector3 d)
        {
            int start = vertices.Count;
            vertices.Add(a);
            vertices.Add(b);
            vertices.Add(c);
            vertices.Add(d);
            AddQuad(triangles, start, start + 1, start + 2, start + 3);
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

        private static Vector3 Average(List<Vector3> points)
        {
            Vector3 sum = Vector3.zero;
            for (int i = 0; i < points.Count; i++)
            {
                sum += points[i];
            }

            return sum / Mathf.Max(1, points.Count);
        }

        private static float Range(System.Random random, float min, float max)
        {
            return Mathf.Lerp(min, max, (float)random.NextDouble());
        }

        private readonly struct PitShape
        {
            public PitShape(List<Vector3> top, List<Vector3> bottom, float depth)
            {
                Top = top;
                Bottom = bottom;
                Depth = depth;
            }

            public readonly List<Vector3> Top;
            public readonly List<Vector3> Bottom;
            public readonly float Depth;
        }
    }
}
