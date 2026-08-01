using System.Collections.Generic;
using UnityEngine;

public class WaypointVehicleMover : MonoBehaviour
{
    [Header("Waypoints")]
    public Transform[] waypoints;

    [Header("Movement")]
    [Min(0.01f)]
    public float speed = 3.0f;

    [Tooltip("How fast the vehicle rotates toward the moving direction.")]
    [Min(1f)]
    public float rotationSpeed = 180f;

    [Tooltip("Teleport back to the first point after reaching the final point.")]
    public bool loopWithTeleport = true;

    [Header("Smooth Corner")]
    [Tooltip("Distance before and after a corner used to create a curved turn.")]
    [Min(0.01f)]
    public float cornerRadius = 3.0f;

    [Tooltip("More samples make the curve smoother.")]
    [Range(3, 100)]
    public int curveSamples = 40;

    [Header("Front Wheel Steering")]
    [Tooltip("Empty parent objects used as steering pivots for front wheels.")]
    public Transform[] frontWheelSteeringPivots;

    [Tooltip("Maximum visual steering angle of front wheels.")]
    [Range(0f, 60f)]
    public float maxSteeringAngle = 30f;

    [Tooltip("How quickly the front wheels visually steer.")]
    [Min(1f)]
    public float steeringSmoothSpeed = 8f;

    [Tooltip("Usually Local Y. Change if your wheel pivot axis is different.")]
    public Vector3 steeringLocalAxis = Vector3.up;

    private readonly List<Vector3> pathPoints = new List<Vector3>();
    private readonly List<float> segmentLengths = new List<float>();

    private int currentSegmentIndex = 0;
    private float currentSteeringAngle = 0f;

    private Quaternion[] frontWheelBaseRotations;

    private void Start()
    {
        BuildPath();

        if (pathPoints.Count >= 2)
        {
            transform.position = pathPoints[0];

            Vector3 initialDir = pathPoints[1] - pathPoints[0];
            if (initialDir.sqrMagnitude > 0.0001f)
            {
                transform.rotation = Quaternion.LookRotation(initialDir.normalized, Vector3.up);
            }
        }

        CacheFrontWheelBaseRotations();
    }

    private void Update()
    {
        if (pathPoints.Count < 2)
            return;

        MoveAlongPath();
    }

    private void BuildPath()
    {
        pathPoints.Clear();
        segmentLengths.Clear();
        currentSegmentIndex = 0;

        if (waypoints == null || waypoints.Length < 2)
            return;

        pathPoints.Add(waypoints[0].position);

        for (int i = 1; i < waypoints.Length - 1; i++)
        {
            Vector3 prev = waypoints[i - 1].position;
            Vector3 corner = waypoints[i].position;
            Vector3 next = waypoints[i + 1].position;

            Vector3 dirIn = (corner - prev).normalized;
            Vector3 dirOut = (next - corner).normalized;

            float distIn = Vector3.Distance(prev, corner);
            float distOut = Vector3.Distance(corner, next);

            float r = Mathf.Min(cornerRadius, distIn * 0.45f, distOut * 0.45f);

            Vector3 curveStart = corner - dirIn * r;
            Vector3 curveEnd = corner + dirOut * r;

            AddPointIfFar(curveStart);

            for (int s = 1; s <= curveSamples; s++)
            {
                float t = s / (float)curveSamples;
                Vector3 curvePoint = QuadraticBezier(curveStart, corner, curveEnd, t);
                AddPointIfFar(curvePoint);
            }
        }

        AddPointIfFar(waypoints[waypoints.Length - 1].position);

        for (int i = 0; i < pathPoints.Count - 1; i++)
        {
            segmentLengths.Add(Vector3.Distance(pathPoints[i], pathPoints[i + 1]));
        }
    }

    private Vector3 QuadraticBezier(Vector3 p0, Vector3 p1, Vector3 p2, float t)
    {
        float u = 1f - t;
        return u * u * p0 + 2f * u * t * p1 + t * t * p2;
    }

    private void AddPointIfFar(Vector3 point)
    {
        if (pathPoints.Count == 0)
        {
            pathPoints.Add(point);
            return;
        }

        if (Vector3.Distance(pathPoints[pathPoints.Count - 1], point) > 0.01f)
        {
            pathPoints.Add(point);
        }
    }

    private void MoveAlongPath()
    {
        if (currentSegmentIndex >= pathPoints.Count - 1)
        {
            if (loopWithTeleport)
            {
                transform.position = pathPoints[0];
                currentSegmentIndex = 0;

                if (pathPoints.Count >= 2)
                {
                    Vector3 resetDir = pathPoints[1] - pathPoints[0];
                    if (resetDir.sqrMagnitude > 0.0001f)
                    {
                        transform.rotation = Quaternion.LookRotation(resetDir.normalized, Vector3.up);
                    }
                }
            }

            return;
        }

        Vector3 target = pathPoints[currentSegmentIndex + 1];
        Vector3 toTarget = target - transform.position;

        if (toTarget.magnitude < 0.05f)
        {
            currentSegmentIndex++;
            return;
        }

        Vector3 moveDir = toTarget.normalized;

        transform.position = Vector3.MoveTowards(
            transform.position,
            target,
            speed * Time.deltaTime
        );

        RotateVehicle(moveDir);
        UpdateFrontWheelSteering(moveDir);
    }

    private void RotateVehicle(Vector3 moveDir)
    {
        if (moveDir.sqrMagnitude < 0.0001f)
            return;

        Quaternion targetRotation = Quaternion.LookRotation(moveDir, Vector3.up);

        transform.rotation = Quaternion.RotateTowards(
            transform.rotation,
            targetRotation,
            rotationSpeed * Time.deltaTime
        );
    }

    private void CacheFrontWheelBaseRotations()
    {
        if (frontWheelSteeringPivots == null)
            return;

        frontWheelBaseRotations = new Quaternion[frontWheelSteeringPivots.Length];

        for (int i = 0; i < frontWheelSteeringPivots.Length; i++)
        {
            if (frontWheelSteeringPivots[i] != null)
            {
                frontWheelBaseRotations[i] = frontWheelSteeringPivots[i].localRotation;
            }
        }
    }

    private void UpdateFrontWheelSteering(Vector3 moveDir)
    {
        if (frontWheelSteeringPivots == null || frontWheelSteeringPivots.Length == 0)
            return;

        float targetSteeringAngle = CalculateLookAheadSteeringAngle();

        currentSteeringAngle = Mathf.Lerp(
            currentSteeringAngle,
            targetSteeringAngle,
            Time.deltaTime * steeringSmoothSpeed
        );

        for (int i = 0; i < frontWheelSteeringPivots.Length; i++)
        {
            if (frontWheelSteeringPivots[i] == null)
                continue;

            Quaternion steeringRotation = Quaternion.AngleAxis(currentSteeringAngle, steeringLocalAxis);
            frontWheelSteeringPivots[i].localRotation = frontWheelBaseRotations[i] * steeringRotation;
        }
    }

    private float CalculateLookAheadSteeringAngle()
    {
        if (pathPoints == null || pathPoints.Count < 2)
            return 0f;

        int lookAheadIndex = Mathf.Min(currentSegmentIndex + 2, pathPoints.Count - 1);

        Vector3 lookAheadDir = pathPoints[lookAheadIndex] - transform.position;
        lookAheadDir.y = 0f;

        if (lookAheadDir.sqrMagnitude < 0.0001f)
            return 0f;

        Vector3 localLookAheadDir = transform.InverseTransformDirection(lookAheadDir.normalized);

        float angle = Mathf.Atan2(localLookAheadDir.x, localLookAheadDir.z) * Mathf.Rad2Deg;

        return Mathf.Clamp(angle, -maxSteeringAngle, maxSteeringAngle);
    }

    private void OnDrawGizmos()
    {
        if (waypoints == null || waypoints.Length < 2)
            return;

        BuildPath();

        Gizmos.color = Color.yellow;

        for (int i = 0; i < pathPoints.Count - 1; i++)
        {
            Gizmos.DrawLine(pathPoints[i], pathPoints[i + 1]);
            Gizmos.DrawSphere(pathPoints[i], 0.12f);
        }

        if (pathPoints.Count > 0)
        {
            Gizmos.DrawSphere(pathPoints[pathPoints.Count - 1], 0.12f);
        }
    }
}