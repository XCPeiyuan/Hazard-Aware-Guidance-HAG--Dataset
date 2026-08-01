using UnityEngine;

public class LoopMoveByPosition : MonoBehaviour
{
    public Vector3 startPosition;
    public Vector3 endPosition;
    public float speed = 2.0f;

    private void Start()
    {
        transform.position = startPosition;
    }

    private void Update()
    {
        transform.position = Vector3.MoveTowards(
            transform.position,
            endPosition,
            speed * Time.deltaTime
        );

        if (Vector3.Distance(transform.position, endPosition) < 0.01f)
        {
            transform.position = startPosition;
        }
    }
}