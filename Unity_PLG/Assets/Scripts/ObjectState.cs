using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class ObjectState : MonoBehaviour
{
    public ColliderState child_collider;
    public BoxCollider boxCollider;
    public Generator generator;

    public Rule rule;
    public List<List<float>> able_rule_edges = new List<List<float>>();
    public List<int> able_rule_indices = new List<int>();

    public bool is_complete = false;
    public float object_index;

    void Awake()
    {
        child_collider = transform.GetChild(0).GetComponent<ColliderState>();

        boxCollider = GetComponent<BoxCollider>();
        boxCollider.size = boxCollider.size + new Vector3(0.5f, 0.5f, 0.5f);

        generator = GameObject.Find("Generator").GetComponent<Generator>();

        string my_name = NameConverter(this.name);
        for (int i = 0; i < generator.gameObjects.Count; i++)
        {
            if (NameConverter(generator.gameObjects[i].name) == my_name)
            {
                object_index = i;
            }
        }

        rule.parent_node = my_name;
        rule.parent_position = transform.position.ToString();
    }

    private string NameConverter(string name)
    {
        if (name.IndexOf(" (") != -1)
            name = name.Substring(0, name.IndexOf(" ("));

        if (name.IndexOf("(Clone)") != -1)
            name = name.Substring(0, name.IndexOf("(Clone)"));
        return name;
    }
    private float GetObjectIndex(string object_name)
    {
        float object_index = -1;
        string my_name = NameConverter(object_name);
        for (int i = 0; i < generator.gameObjects.Count; i++)
        {
            if (NameConverter(generator.gameObjects[i].name) == my_name)
            {
                object_index = i;
            }
        }

        return object_index;
    }

    public bool CheckAbleRule()
    {
        able_rule_indices = new List<int>();
        for (int i = 0; i < generator.rules_list.Count; i++)
        {
            bool is_able_rule = true;
            for(int j = 0; j < rule.child_nodes.Count; j++)
            {
                bool is_found = false;
                for(int k = 0; k < generator.rules_list[i].child_nodes.Count; k++)
                {
                    if(rule.child_nodes[j] == generator.rules_list[i].child_nodes[k] && rule.child_directions[j] == generator.rules_list[i].child_directions[k])
                    {
                        is_found = true;
                    }
                }

                if (!is_found)
                {
                    is_able_rule = false;
                }
            }

            if (is_able_rule)
            {
                able_rule_indices.Add(i);
            }
        }
        if(able_rule_indices.Count > 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    public void FindAbleRules()
    {
        rule.parent_position = transform.position.ToString();

        able_rule_edges = new List<List<float>>();
        for (int i = 0; i < generator.rules_list.Count; i++)
        {
            Rule real_rule = generator.rules_list[i];
            if (real_rule.parent_node == rule.parent_node)
            {
                for(int j = 0; j < real_rule.child_nodes.Count; j++)
                {
                    for(int k = 0; k < rule.child_nodes.Count; k++)
                    {
                        if(rule.child_nodes[k] == real_rule.child_nodes[j] && rule.child_directions[k] == real_rule.child_directions[j])
                        {
                            continue;
                        }

                        float parent_index = GetObjectIndex(rule.parent_node);
                        float parent_pos_x = StringToVector3(rule.parent_position).x;
                        float parent_pos_y = StringToVector3(rule.parent_position).y;
                        float parent_pos_z = StringToVector3(rule.parent_position).z;

                        float child_index = GetObjectIndex(real_rule.child_nodes[j]);
                        float child_pos_x = StringToVector3(real_rule.child_positions[j]).x + parent_pos_x;
                        float child_pos_y = StringToVector3(real_rule.child_positions[j]).y + parent_pos_y;
                        float child_pos_z = StringToVector3(real_rule.child_positions[j]).z + parent_pos_z;

                        List<float> floatList = new List<float>(new float[] {
                            parent_index, parent_pos_x, parent_pos_y, parent_pos_z,
                            child_index, child_pos_x, child_pos_y, child_pos_z });

                        if(!able_rule_edges.Exists(innerList => innerList.SequenceEqual(floatList)))
                        {
                            able_rule_edges.Add(floatList);
                        }
                        string combinedString = string.Join(" ", floatList);
                    }
                }
            }
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if(other.gameObject.tag != "Object")
        {
            return;
        }
        string my_name = NameConverter(name);
        string other_name = NameConverter(other.name);

        Vector3 direction = other.gameObject.transform.position - transform.position;
        direction = direction / direction.magnitude;

        if(float.IsNaN(direction.x) && float.IsNaN(direction.y) && float.IsNaN(direction.z))
        {
            direction = Vector3.zero;
        }

        rule.child_nodes.Add(other_name);
        rule.child_directions.Add(direction.ToString());
        rule.child_positions.Add(other.transform.position.ToString());
    }

    public static Vector3 StringToVector3(string sVector)
    {
        // Remove the parentheses
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        // split the items
        string[] sArray = sVector.Split(',');

        // store as a Vector3
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }
}
