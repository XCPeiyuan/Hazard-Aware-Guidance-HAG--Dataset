using UnityEngine;

namespace ProceduralManholeGenerator
{
    public enum ProceduralManholeMaterialKind
    {
        Concrete,
        CastIronCover,
        DarkOpening,
        Pipe,
        Ground
    }

    public static class ProceduralManholeMaterials
    {
        public static Material Create(ProceduralManholeMaterialKind kind, int seed, int textureSize, float variation)
        {
            int size = Mathf.Clamp(textureSize, 64, 1024);
            variation = Mathf.Clamp01(variation);

            switch (kind)
            {
                case ProceduralManholeMaterialKind.CastIronCover:
                    return CreateMaterial("PM_Auto_CastIronCover", GenerateCastIronTexture(seed, size, variation), 0.85f, 0.48f);
                case ProceduralManholeMaterialKind.DarkOpening:
                    return CreateMaterial("PM_Auto_DarkOpening", GenerateDarkOpeningTexture(seed, size, variation), 0f, 0.02f);
                case ProceduralManholeMaterialKind.Pipe:
                    return CreateMaterial("PM_Auto_Pipe", GeneratePipeTexture(seed, size, variation), 0f, 0.18f);
                case ProceduralManholeMaterialKind.Ground:
                    return CreateMaterial("PM_Auto_Ground", GenerateGroundTexture(seed, size, variation), 0f, 0.34f);
                default:
                    return CreateMaterial("PM_Auto_Concrete", GenerateConcreteTexture(seed, size, variation), 0f, 0.28f);
            }
        }

        private static Material CreateMaterial(string name, Texture2D texture, float metallic, float smoothness)
        {
            Shader shader = FindSupportedShader();
            Material material = new Material(shader) { name = name };
            ApplyBaseColor(material, Color.white);
            ApplyBaseTexture(material, texture);

            if (material.HasProperty("_Metallic"))
            {
                material.SetFloat("_Metallic", metallic);
            }

            if (material.HasProperty("_Glossiness"))
            {
                material.SetFloat("_Glossiness", smoothness);
            }

            if (material.HasProperty("_Smoothness"))
            {
                material.SetFloat("_Smoothness", smoothness);
            }

            return material;
        }

        private static Texture2D GenerateConcreteTexture(int seed, int size, float variation)
        {
            Texture2D texture = NewTexture("PM_Auto_Concrete_Texture", size);
            Color baseColor = Jitter(new Color(0.58f, 0.57f, 0.52f), seed, variation * 0.12f);

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size;
                    float v = (float)y / size;
                    float grain = FractalNoise(u, v, seed, 7f, 4);
                    float pores = Mathf.PerlinNoise(u * 58f + seed * 0.13f, v * 58f + seed * 0.07f);
                    float stain = Mathf.PerlinNoise(u * 2.1f + seed * 0.01f, v * 2.1f + seed * 0.02f);
                    float value = 0.96f + grain * 0.18f - Mathf.Max(0f, stain - 0.72f) * 0.22f - Mathf.Max(0f, pores - 0.9f) * 0.16f;
                    texture.SetPixel(x, y, ClampColor(baseColor * value));
                }
            }

            texture.Apply();
            return texture;
        }

        private static Texture2D GenerateCastIronTexture(int seed, int size, float variation)
        {
            Texture2D texture = NewTexture("PM_Auto_CastIronCover_Texture", size);
            Color iron = Jitter(new Color(0.28f, 0.27f, 0.25f), seed, variation * 0.1f);
            Color rust = Jitter(new Color(0.42f, 0.16f, 0.055f), seed + 8, variation * 0.18f);

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size;
                    float v = (float)y / size;
                    float dx = u - 0.5f;
                    float dy = v - 0.5f;
                    float radius = Mathf.Sqrt(dx * dx + dy * dy);
                    float angle = Mathf.Atan2(dy, dx);
                    float ring = Mathf.Abs(Mathf.Sin(radius * 92f));
                    float ribs = Mathf.Abs(Mathf.Sin(angle * 12f));
                    float rustMask = Mathf.PerlinNoise(u * 6f + seed * 0.03f, v * 6f + seed * 0.05f);
                    float noise = FractalNoise(u, v, seed, 18f, 3);
                    Color color = iron * (0.78f + noise * 0.38f + ring * 0.12f + ribs * 0.08f);

                    if (rustMask > 0.72f)
                    {
                        color = Color.Lerp(color, rust, (rustMask - 0.72f) * 2.6f * variation);
                    }

                    texture.SetPixel(x, y, ClampColor(color));
                }
            }

            texture.Apply();
            return texture;
        }

        private static Texture2D GenerateDarkOpeningTexture(int seed, int size, float variation)
        {
            Texture2D texture = NewTexture("PM_Auto_DarkOpening_Texture", size);
            Color edge = new Color(0.08f, 0.075f, 0.065f);
            Color center = new Color(0.006f, 0.005f, 0.004f);

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size;
                    float v = (float)y / size;
                    float dx = u - 0.5f;
                    float dy = v - 0.5f;
                    float radius = Mathf.Clamp01(Mathf.Sqrt(dx * dx + dy * dy) * 2f);
                    float noise = FractalNoise(u, v, seed, 5f, 3) * variation;
                    Color color = Color.Lerp(center, edge, Mathf.Pow(radius, 2.2f)) * (0.88f + noise * 0.2f);
                    texture.SetPixel(x, y, ClampColor(color));
                }
            }

            texture.Apply();
            return texture;
        }

        private static Texture2D GeneratePipeTexture(int seed, int size, float variation)
        {
            Texture2D texture = NewTexture("PM_Auto_Pipe_Texture", size);
            Color baseColor = Jitter(new Color(0.36f, 0.36f, 0.34f), seed, variation * 0.12f);

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size;
                    float v = (float)y / size;
                    float longitudinal = Mathf.Abs(Mathf.Sin(v * 36f));
                    float grime = Mathf.PerlinNoise(u * 8f + seed * 0.02f, v * 4f + seed * 0.04f);
                    float value = 0.86f + FractalNoise(u, v, seed, 11f, 3) * 0.28f - Mathf.Max(0f, grime - 0.68f) * 0.25f + longitudinal * 0.05f;
                    texture.SetPixel(x, y, ClampColor(baseColor * value));
                }
            }

            texture.Apply();
            return texture;
        }

        private static Texture2D GenerateGroundTexture(int seed, int size, float variation)
        {
            Texture2D texture = NewTexture("PM_Auto_Ground_Texture", size);
            Color asphalt = Jitter(new Color(0.48f, 0.46f, 0.41f), seed, variation * 0.1f);

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size;
                    float v = (float)y / size;
                    float aggregate = FractalNoise(u, v, seed, 28f, 4);
                    float broadStain = Mathf.PerlinNoise(u * 3f + seed * 0.02f, v * 3f + seed * 0.04f);
                    float value = 0.94f + aggregate * 0.22f - Mathf.Max(0f, broadStain - 0.74f) * 0.14f;
                    texture.SetPixel(x, y, ClampColor(asphalt * value));
                }
            }

            texture.Apply();
            return texture;
        }

        private static Texture2D NewTexture(string name, int size)
        {
            Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, true)
            {
                name = name,
                wrapMode = TextureWrapMode.Repeat,
                filterMode = FilterMode.Bilinear
            };
            return texture;
        }

        private static float FractalNoise(float u, float v, int seed, float scale, int octaves)
        {
            float value = 0f;
            float amplitude = 0.5f;
            float frequency = 1f;
            float total = 0f;

            for (int i = 0; i < octaves; i++)
            {
                value += Mathf.PerlinNoise(u * scale * frequency + seed * 0.031f, v * scale * frequency + seed * 0.047f) * amplitude;
                total += amplitude;
                amplitude *= 0.5f;
                frequency *= 2f;
            }

            return total > 0f ? value / total : 0f;
        }

        private static Color Jitter(Color color, int seed, float amount)
        {
            float r = Mathf.PerlinNoise(seed * 0.11f, 0.17f) * 2f - 1f;
            float g = Mathf.PerlinNoise(seed * 0.13f, 0.31f) * 2f - 1f;
            float b = Mathf.PerlinNoise(seed * 0.19f, 0.53f) * 2f - 1f;
            return ClampColor(new Color(color.r + r * amount, color.g + g * amount, color.b + b * amount, 1f));
        }

        private static Color ClampColor(Color color)
        {
            return new Color(
                Mathf.Clamp01(color.r),
                Mathf.Clamp01(color.g),
                Mathf.Clamp01(color.b),
                Mathf.Clamp01(color.a));
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

        private static void ApplyBaseColor(Material material, Color color)
        {
            if (material.HasProperty("_BaseColor"))
            {
                material.SetColor("_BaseColor", color);
            }

            if (material.HasProperty("_Color"))
            {
                material.SetColor("_Color", color);
            }
        }

        private static void ApplyBaseTexture(Material material, Texture2D texture)
        {
            if (material.HasProperty("_BaseMap"))
            {
                material.SetTexture("_BaseMap", texture);
            }

            if (material.HasProperty("_MainTex"))
            {
                material.SetTexture("_MainTex", texture);
            }

            material.mainTexture = texture;
        }
    }
}
