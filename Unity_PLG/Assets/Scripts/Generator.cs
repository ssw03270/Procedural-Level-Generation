using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Text;
using Newtonsoft.Json;
using System.Linq;

public class Generator : Agent
{
    public List<GameObject> gameObjects = new List<GameObject>();
    public List<List<float>> level_edge_list = new List<List<float>>();
    public List<List<List<float>>> rule_with_edge_list = new List<List<List<float>>>();
    public List<Rule> rules_list = new List<Rule>();
    public List<string> unique_rule = new List<string>();
    public List<List<float>> able_rules = new List<List<float>>();
    public List<string> able_rules_string = new List<string>();

    private string output_path = "C:\\Users\\Dobby\\Documents\\GitHub\\Procedural-Level-Generation\\Rules";

    private void Start()
    {
        if (Directory.Exists(output_path))
        {
            string[] file_paths = Directory.GetFiles(output_path);

            foreach (string path in file_paths)
            {
                FileStream fileStream = new FileStream(path, FileMode.Open); // 경로에 있는 파일을 열어주고,
                byte[] data = new byte[fileStream.Length];
                fileStream.Read(data, 0, data.Length);
                fileStream.Close();

                string json = Encoding.UTF8.GetString(data);

                if (!unique_rule.Exists(innerList => innerList.SequenceEqual(json)))
                {
                    unique_rule.Add(json);
                }
                else
                {
                    continue;
                }

                Rule rule_json = JsonUtility.FromJson<Rule>(json);
                List<List<float>> rule_list = new List<List<float>>();

                for (int i = 0; i < rule_json.child_nodes.Count; i++)
                {
                    List<float> edge = new List<float>();

                    float parent_index = -1;
                    for (int j = 0; j < gameObjects.Count; j++)
                    {
                        if (gameObjects[j].name == rule_json.parent_node)
                        {
                            parent_index = j;
                            break;
                        }
                    }

                    Vector3 parent_position = StringToVector3(rule_json.parent_position);

                    float child_index = -1;
                    for (int j = 0; j < gameObjects.Count; j++)
                    {
                        if (gameObjects[j].name == rule_json.child_nodes[i])
                        {
                            child_index = j;
                            break;
                        }
                    }

                    Vector3 child_positions = StringToVector3(rule_json.child_positions[i]);

                    edge = new List<float>(new float[] { parent_index, parent_position.x, parent_position.y, parent_position.z,
                    child_index, child_positions.x, child_positions.y, child_positions.z});
                    rule_list.Add(new List<float>(edge));
                }

                rule_with_edge_list.Add(rule_list);
                rules_list.Add(rule_json);
            }
        }
    }
    public override void OnEpisodeBegin()
    {
        GameObject[] all_objects = GameObject.FindGameObjectsWithTag("Object");
        foreach (GameObject gameObject in all_objects)
        {
            Destroy(gameObject);
            if(gameObject != null)
            {
                gameObject.GetComponent<ObjectState>().boxCollider = gameObject.GetComponent<BoxCollider>();
                gameObject.GetComponent<ObjectState>().boxCollider.enabled = false;
                if(gameObject.GetComponent<ObjectState>().child_collider != null)
                {
                    gameObject.GetComponent<ObjectState>().child_collider.boxCollider = gameObject.GetComponent<ObjectState>().child_collider.GetComponent<BoxCollider>();
                    gameObject.GetComponent<ObjectState>().child_collider.boxCollider.enabled = false;
                }
            }
        }
        level_edge_list = new List<List<float>>();

        all_objects = GameObject.FindGameObjectsWithTag("Object");

        List<List<float>> init_rules = rule_with_edge_list[Random.Range(0, rule_with_edge_list.Count)];
        int idx = Random.Range(0, init_rules.Count);
        List<float> init_rule = init_rules[idx];
        ApplyRule(init_rule, true);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 레벨 정보 찾기
        level_edge_list = new List<List<float>>();
        GameObject[] all_objects = GameObject.FindGameObjectsWithTag("Object");
        foreach (GameObject gameObject in all_objects)
        {
            float parent_index = GetObjectIndex(gameObject.GetComponent<ObjectState>().rule.parent_node);
            float parent_pos_x = StringToVector3(gameObject.GetComponent<ObjectState>().rule.parent_position).x;
            float parent_pos_y = StringToVector3(gameObject.GetComponent<ObjectState>().rule.parent_position).y;
            float parent_pos_z = StringToVector3(gameObject.GetComponent<ObjectState>().rule.parent_position).z;

            for(int i = 0; i < gameObject.GetComponent<ObjectState>().rule.child_nodes.Count; i++)
            {
                float child_index = GetObjectIndex(gameObject.GetComponent<ObjectState>().rule.child_nodes[i]);
                float child_pos_x = StringToVector3(gameObject.GetComponent<ObjectState>().rule.child_positions[i]).x;
                float child_pos_y = StringToVector3(gameObject.GetComponent<ObjectState>().rule.child_positions[i]).y;
                float child_pos_z = StringToVector3(gameObject.GetComponent<ObjectState>().rule.child_positions[i]).z;
                
                level_edge_list.Add(new List<float>(new float[] { 
                    parent_index, parent_pos_x, parent_pos_y, parent_pos_z,
                    child_index, child_pos_x, child_pos_y, child_pos_z }));
            }
        }
        print(level_edge_list.Count);
        // level 정보 입력으로 넣기
        foreach (List<float> edge in level_edge_list)
        {
            sensor.AddObservation(edge);
        }
        for(int i = level_edge_list.Count; i < 100; i++)
        {
            sensor.AddObservation(new List<float>(new float[] { 0, 0, 0, 0, 0, 0, 0, 0}));
        }

        // 가능한 rule 찾기
        all_objects = GameObject.FindGameObjectsWithTag("Object");
        able_rules = new List<List<float>>();
        foreach (GameObject gameObject in all_objects)
        {
            if (!gameObject.GetComponent<ObjectState>().CheckAbleRule())
            {
                continue;
            }
            else
            {
                gameObject.GetComponent<ObjectState>().FindAbleRules();
                able_rules = gameObject.GetComponent<ObjectState>().able_rule_edges;

                break;
            }
        }
        // rule 정보 입력으로 넣기
        for(int i = 0; i < able_rules.Count; i++)
        {
            sensor.AddObservation(able_rules[i]);
        }

        for(int i = able_rules.Count; i < 100;  i++)
        {
            sensor.AddObservation(new List<float>(new float[] { 0, 0, 0, 0, 0, 0, 0, 0 }));
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // 1 이면, 다음 오브젝트로 넘어가기
        int is_complete = actions.DiscreteActions[0];

        // 가능한 rule 찾기
        GameObject[] all_objects = GameObject.FindGameObjectsWithTag("Object");
        able_rules = new List<List<float>>();
        able_rules_string = new List<string>();
        foreach (GameObject gameObject in all_objects)
        {
            if (gameObject.GetComponent<ObjectState>().is_complete)
            {
                continue;
            }
            else
            {
                gameObject.GetComponent<ObjectState>().FindAbleRules();
                for (int i = 0; i < gameObject.GetComponent<ObjectState>().able_rule_edges.Count; i++)
                {
                    if (!able_rules.Exists(innerList => innerList.SequenceEqual(gameObject.GetComponent<ObjectState>().able_rule_edges[i])))
                    {
                        able_rules.Add(gameObject.GetComponent<ObjectState>().able_rule_edges[i]);
                        able_rules_string.Add(string.Join(" ", gameObject.GetComponent<ObjectState>().able_rule_edges[i]));
                    }
                }
                if(is_complete == 1)
                {
                    gameObject.GetComponent<ObjectState>().is_complete = true;
                }

                break;
            }
        }
        if (able_rules.Count > 0)
        {
            int rule_index = actions.DiscreteActions[1];
            if(rule_index >= able_rules.Count)
            {
                rule_index = able_rules.Count - 1;
            }
            List<float> rule = able_rules[rule_index];

            ApplyRule(rule, false);

            AddReward(1);

/*            foreach (GameObject gameObject in all_objects)
            {
                if (!gameObject.GetComponent<ObjectState>().CheckAbleRule())
                {
                    AddReward(-1.0f);
                }
            }*/


            int unable_object_count = 0;
            foreach (GameObject gameObject in all_objects)
            {
                if (!gameObject.GetComponent<ObjectState>().CheckAbleRule())
                {
                    unable_object_count += 1;
                }
            }
            AddReward(unable_object_count / all_objects.Length);

            all_objects = GameObject.FindGameObjectsWithTag("Collider");
            foreach (GameObject gameObject in all_objects)
            {
                if (gameObject.GetComponent<ColliderState>().is_overlap)
                {
                    AddReward(-10.0f);
                    EndEpisode();
                }
            }
        }
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

    public void ApplyRule(List<float> rule, bool is_init)
    {
        // 만약 최초 생성이라면
        if (is_init)
        {
            float parent_index = rule[0];
            Vector3 parent_position = new Vector3(rule[1], rule[2], rule[3]);

            GameObject parent_object = Instantiate(gameObjects[(int)parent_index], parent_position, gameObjects[(int)parent_index].transform.rotation);

            parent_object.GetComponent<ObjectState>().object_index = parent_index;
            parent_object.transform.position = parent_position;

            float child_index = rule[4];
            Vector3 child_position = new Vector3(rule[5], rule[6], rule[7]);

            GameObject child_object = Instantiate(gameObjects[(int)child_index], child_position, gameObjects[(int)child_index].transform.rotation);
            child_object.GetComponent<ObjectState>().object_index = child_index;
            child_object.transform.position = child_position;
        }
        else
        {
            Vector3 parent_position = new Vector3(rule[1], rule[2], rule[3]);

            float child_index = rule[4];
            Vector3 child_position = new Vector3(rule[5], rule[6], rule[7]);

            GameObject child_object = Instantiate(gameObjects[(int)child_index], child_position, gameObjects[(int)child_index].transform.rotation);
            child_object.GetComponent<ObjectState>().object_index = child_index;
            child_object.transform.position = child_position;
        }
    }

    private float GetObjectIndex(string object_name)
    {
        float object_index = -1;
        string my_name = NameConverter(object_name);
        for (int i = 0; i < gameObjects.Count; i++)
        {
            if (NameConverter(gameObjects[i].name) == my_name)
            {
                object_index = i;
            }
        }

        return object_index;
    }
    private string NameConverter(string name)
    {
        if (name.IndexOf(" (") != -1)
            name = name.Substring(0, name.IndexOf(" ("));

        if (name.IndexOf("(Clone)") != -1)
            name = name.Substring(0, name.IndexOf("(Clone)"));
        return name;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actions = actionsOut.DiscreteActions;
        actions[0] = Random.Range(0, 2);
        actions[1] = Random.Range(0, 100);
    }
}
