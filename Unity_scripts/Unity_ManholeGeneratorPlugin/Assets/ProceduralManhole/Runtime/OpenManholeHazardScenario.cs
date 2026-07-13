using UnityEngine;

namespace ProceduralManholeGenerator
{
    public enum CoverState
    {
        Absent,
        Displaced,
        Partial
    }

    public enum CoverDamage
    {
        None
    }

    [ExecuteAlways]
    [DisallowMultipleComponent]
    public class OpenManholeHazardScenario : MonoBehaviour
    {
        private const string CoverName = "PM_Detachable_Cover";

        [Header("Generated Manhole")]
        public ProceduralManhole manhole;
        public bool autoCreateManhole = true;

        [Header("Hazard State")]
        public CoverState coverState = CoverState.Displaced;
        public CoverDamage coverDamage = CoverDamage.None;

        [Header("Partial Cover State")]
        [Range(0f, 1f)] public float partialOpenAmount = 0.55f;
        [Range(0f, 360f)] public float partialDirectionDegrees = 35f;

        [Header("Displaced Cover State")]
        [Min(0f)] public float displacedDistance = 2.3f;
        [Range(0f, 360f)] public float displacedDirectionDegrees = 55f;

        [Header("Cover Pose Variation")]
        [Range(-180f, 180f)] public float coverYawDegrees = 18f;
        [Range(0f, 75f)] public float coverTiltDegrees = 8f;

        [Header("Ground Cutout Surface")]
        public bool generateGroundCutoutSurface = true;
        public Vector2 groundSize = new Vector2(6f, 6f);
        [Min(0f)] public float groundOpeningClearance = 0.04f;
        public Material groundMaterial;

        public void GenerateScenario()
        {
            EnsureManhole();

            if (manhole == null)
            {
                Debug.LogWarning("OpenManholeHazardScenario could not find or create a ProceduralManhole.", this);
                return;
            }

            manhole.generateCover = true;
            manhole.coverStartsRemoved = false;
            ApplyGroundSettingsToManhole();
            manhole.Generate();
            ApplyCoverState();
        }

        public void ApplyCoverState()
        {
            EnsureManhole();

            if (manhole == null)
            {
                return;
            }

            Transform cover = FindCover();
            if (cover == null)
            {
                return;
            }

            Vector3 basePosition = GetClosedCoverPosition();

            switch (coverState)
            {
                case CoverState.Absent:
                    cover.gameObject.SetActive(false);
                    cover.localPosition = basePosition;
                    cover.localRotation = Quaternion.identity;
                    break;
                case CoverState.Displaced:
                    cover.gameObject.SetActive(true);
                    ApplyCoverPose(cover, basePosition, displacedDirectionDegrees, displacedDistance, coverTiltDegrees);
                    break;
                case CoverState.Partial:
                    cover.gameObject.SetActive(true);
                    float distance = GetFullSlideDistance(partialDirectionDegrees) * Mathf.Clamp01(partialOpenAmount);
                    ApplyCoverPose(cover, basePosition, partialDirectionDegrees, distance, coverTiltDegrees);
                    break;
            }
        }

        private void EnsureManhole()
        {
            if (manhole != null)
            {
                return;
            }

            manhole = GetComponentInChildren<ProceduralManhole>();

            if (manhole == null && autoCreateManhole)
            {
                GameObject child = new GameObject("Procedural Manhole Assembly");
                child.transform.SetParent(transform, false);
                manhole = child.AddComponent<ProceduralManhole>();
            }
        }

        private Transform FindCover()
        {
            Transform root = manhole.transform;
            for (int i = 0; i < root.childCount; i++)
            {
                Transform child = root.GetChild(i);
                if (child.name == CoverName)
                {
                    return child;
                }
            }

            return null;
        }

        private void ApplyGroundSettingsToManhole()
        {
            manhole.generateGroundCutoutSurface = generateGroundCutoutSurface;
            manhole.groundSize = new Vector2(Mathf.Max(1f, groundSize.x), Mathf.Max(1f, groundSize.y));
            manhole.groundOpeningClearance = Mathf.Max(0f, groundOpeningClearance);

            if (groundMaterial != null)
            {
                manhole.groundMaterial = groundMaterial;
            }
        }

        private Vector3 GetClosedCoverPosition()
        {
            return new Vector3(0f, manhole.GetCoverRestCenterY(), 0f);
        }

        private void ApplyCoverPose(Transform cover, Vector3 basePosition, float directionDegrees, float distance, float tiltDegrees)
        {
            Vector3 direction = DirectionFromDegrees(directionDegrees);
            cover.localPosition = basePosition + direction * Mathf.Max(0f, distance);

            Quaternion yaw = Quaternion.AngleAxis(coverYawDegrees + directionDegrees, Vector3.up);
            Quaternion tilt = Quaternion.AngleAxis(tiltDegrees, Vector3.right);
            cover.localRotation = yaw * tilt;
        }

        private float GetFullSlideDistance(float directionDegrees)
        {
            Vector3 direction = DirectionFromDegrees(directionDegrees);

            if (manhole.shape == ManholeShape.Circular)
            {
                return manhole.openingRadius * 2f + manhole.coverClearance;
            }

            Vector2 half = manhole.openingSize * 0.5f;
            float x = Mathf.Abs(direction.x);
            float z = Mathf.Abs(direction.z);
            float openingBoundary = Mathf.Min(
                x > 0.0001f ? half.x / x : float.PositiveInfinity,
                z > 0.0001f ? half.y / z : float.PositiveInfinity);

            if (float.IsInfinity(openingBoundary))
            {
                openingBoundary = Mathf.Max(half.x, half.y);
            }

            return openingBoundary * 2f + manhole.coverClearance;
        }

        private static Vector3 DirectionFromDegrees(float degrees)
        {
            float radians = degrees * Mathf.Deg2Rad;
            return new Vector3(Mathf.Cos(radians), 0f, Mathf.Sin(radians)).normalized;
        }
    }
}
