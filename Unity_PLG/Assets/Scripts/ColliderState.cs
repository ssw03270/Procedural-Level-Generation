using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ColliderState : MonoBehaviour
{
    public bool is_overlap = false;
    public BoxCollider boxCollider;
    // Start is called before the first frame update
    void Awake()
    {
        boxCollider = GetComponent<BoxCollider>();
        boxCollider.size = boxCollider.size - new Vector3(0.5f, 0.5f, 0.5f);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.tag != "Collider")
        {
            return;
        }

        is_overlap = true;
    }
}
