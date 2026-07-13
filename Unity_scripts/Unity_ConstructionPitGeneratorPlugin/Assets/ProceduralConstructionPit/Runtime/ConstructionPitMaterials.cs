using UnityEngine;

namespace ProceduralConstructionPitGenerator
{
    public enum ConstructionPitMaterialKind
    {
        Ground,
        Soil,
        DarkPit,
        Debris
    }

    public static class ConstructionPitMaterials
    {
        public static Material Create(ConstructionPitMaterialKind kind, int seed, int textureSize, float variation)
        {
            int size = Mathf.Clamp(textureSize, 64, 1024);
            variation = Mathf.Clamp01(variation);

            switch (kind)
            {
                case ConstructionPitMaterialKind.Soil:
                    return CreateMaterial("PCP_Auto_Soil", GenerateTexture(seed, size, new Color(0.42f, 0.31f, 0.2f), variation, 18f), 0f, 0.22f);
                case ConstructionPitMaterialKind.DarkPit:
                    return CreateMaterial("PCP_Auto_DarkPit", GenerateTexture(seed, size, new Color(0.08f, 0.055f, 0.035f), variation, 7f), 0f, 0.08f);
                case ConstructionPitMaterialKind.Debris:
                    return CreateMaterial("PCP_Auto_Debris", GenerateTexture(seed, size, new Color(0.38f, 0.34f, 0.28f), variation, 24f), 0f, 0.18f);
                default:
                    return CreateMaterial("PCP_Auto_Ground", GenerateTexture(seed, size, new Color(0.44f, 0.42f, 0.38f), variation, 30f), 0f, 0.3f);
            }
        }

        public static Shader FindSupportedShader()
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader != null) return shader;
            shader = Shader.Find("HDRP/Lit");
            if (shader != null) return shader;
            shader = Shader.Find("Standard");
            if (shader != null) return shader;
            return Shader.Find("Sprites/Default");
        }

        private static Material CreateMaterial(string name, Texture2D texture, float metallic, float smoothness)
        {
            Material material = new Material(FindSupportedShader()) { name = name };
            SetColor(material, Color.white);
            SetTexture(material, texture);

            if (material.HasProperty("_Metallic")) material.SetFloat("_Metallic", metallic);
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", smoothness);
            if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", smoothness);

            return material;
        }

        private static Texture2D GenerateTexture(int seed, int size, Color baseColor, float variation, float scale)
        {
            Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, true)
            {
                name = "PCP_Auto_Texture",
                wrapMode = TextureWrapMode.Repeat,
                filterMode = FilterMode.Bilinear
            };

            Color color = Jitter(baseColor, seed, variation * 0.12f);
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size;
                    float v = (float)y / size;
                    float grain = FractalNoise(u, v, seed, scale, 4);
                    float stain = Mathf.PerlinNoise(u * 3f + seed * 0.03f, v * 3f + seed * 0.04f);
                    float value = 0.82f + grain * 0.36f - Mathf.Max(0f, stain - 0.72f) * 0.2f;
                    texture.SetPixel(x, y, Clamp(color * value));
                }
            }

            texture.Apply();
            return texture;
        }

        private static void SetColor(Material material, Color color)
        {
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
            if (material.HasProperty("_Color")) material.SetColor("_Color", color);
        }

        private static void SetTexture(Material material, Texture2D texture)
        {
            if (material.HasProperty("_BaseMap")) material.SetTexture("_BaseMap", texture);
            if (material.HasProperty("_MainTex")) material.SetTexture("_MainTex", texture);
            material.mainTexture = texture;
        }

        private static float FractalNoise(float u, float v, int seed, float scale, int octaves)
        {
            float value = 0f;
            float amplitude = 0.5f;
            float total = 0f;
            float frequency = 1f;

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
            return Clamp(new Color(color.r + r * amount, color.g + g * amount, color.b + b * amount, 1f));
        }

        private static Color Clamp(Color color)
        {
            return new Color(Mathf.Clamp01(color.r), Mathf.Clamp01(color.g), Mathf.Clamp01(color.b), Mathf.Clamp01(color.a));
        }
    }
}
