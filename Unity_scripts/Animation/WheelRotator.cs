using UnityEngine;

public class WheelRotator : MonoBehaviour
{
    public enum RotationAxis
    {
        LocalX,
        LocalY,
        LocalZ
    }

    [Header("Wheel Rotation")]
    public RotationAxis rotationAxis = RotationAxis.LocalX;

    [Tooltip("Rotation speed in degrees per second.")]
    public float rotationSpeed = 360f;

    [Tooltip("Reverse wheel rotation direction.")]
    public bool reverse = false;

    [Header("Optional: Auto calculate speed from parent movement")]
    public bool useParentMovementSpeed = false;

    [Tooltip("Wheel radius in meters. Used only when useParentMovementSpeed is true.")]
    public float wheelRadius = 0.35f;

    [Tooltip("The moving parent object. If empty, this script will use transform.root.")]
    public Transform movingParent;

    private Vector3 lastParentPosition;
    private bool hasLastPosition = false;

    private void Start()
    {
        if (movingParent == null)
        {
            movingParent = transform.root;
        }

        if (movingParent != null)
        {
            lastParentPosition = movingParent.position;
            hasLastPosition = true;
        }
    }

    private void Update()
    {
        float finalSpeed = rotationSpeed;

        if (useParentMovementSpeed && movingParent != null && hasLastPosition)
        {
            float moveDistance = Vector3.Distance(movingParent.position, lastParentPosition);

            if (Time.deltaTime > 0.0001f && wheelRadius > 0.0001f)
            {
                float linearSpeed = moveDistance / Time.deltaTime;
                float angularSpeedRad = linearSpeed / wheelRadius;
                finalSpeed = angularSpeedRad * Mathf.Rad2Deg;
            }

            lastParentPosition = movingParent.position;
        }

        if (reverse)
        {
            finalSpeed = -finalSpeed;
        }

        Vector3 axis = GetLocalAxis(rotationAxis);

        transform.Rotate(axis, finalSpeed * Time.deltaTime, Space.Self);
    }

    private Vector3 GetLocalAxis(RotationAxis axis)
    {
        switch (axis)
        {
            case RotationAxis.LocalX:
                return Vector3.right;
            case RotationAxis.LocalY:
                return Vector3.up;
            case RotationAxis.LocalZ:
                return Vector3.forward;
            default:
                return Vector3.right;
        }
    }
}