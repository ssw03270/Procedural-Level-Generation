using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;
using System.Linq;

[Serializable]
public class Rule
{
    public string parent_node;
    public string parent_position;
    public string parent_scale;
    public List<string> child_nodes = new List<string>();
    public List<string> child_directions = new List<string>();
    public List<string> child_positions = new List<string>();
    public List<string> child_scales = new List<string>();

    public void SortLists()
    {
        var sorted = child_nodes
            .Select((node, index) => new
            {
                Node = node,
                Direction = child_directions[index],
                Position = child_positions[index],
                Scale = child_scales[index]
            })
            .OrderBy(x => x.Node)
            .ThenBy(x => x.Direction)
            .ThenBy(x => x.Position)
            .ThenBy(x => x.Scale)
            .ToList();

        child_nodes = sorted.Select(x => x.Node).ToList();
        child_directions = sorted.Select(x => x.Direction).ToList();
        child_positions = sorted.Select(x => x.Position).ToList();
        child_scales = sorted.Select(x => x.Scale).ToList();
    }
    public override bool Equals(object obj)
    {
        if (obj == null || GetType() != obj.GetType())
            return false;

        Rule other = (Rule)obj;

        return parent_node == other.parent_node
            && child_nodes.SequenceEqual(other.child_nodes)
            && child_directions.SequenceEqual(other.child_directions)
            && child_scales.SequenceEqual(other.child_scales);
    }

    /*public override int GetHashCode()
    {
        // Using XOR for combining hash codes. You might want to use a better approach for larger applications.
        return parent_node.GetHashCode()
            ^ child_nodes.GetHashCode()
            ^ child_directions.GetHashCode()
            ^ child_directions.GetHashCode();
    }*/

}

public class GetNeighborObject : MonoBehaviour
{
    private BoxCollider boxCollider;

    private List<string> edge_list = new List<string>();

    private string output_path = "Assets/Resources/Rules/";

    private Rule rule = new Rule();

    // Start is called before the first frame update
    void Awake()
    {
        if (Directory.Exists(output_path))
        {
            string[] fhile_paths = Directory.GetFiles(output_path);

            foreach (string path in fhile_paths)
            {
                File.Delete(path);
            }
        }

        boxCollider = GetComponent<BoxCollider>();
        boxCollider.size = boxCollider.size + new Vector3(0.5f, 0.5f, 0.5f);

        string my_name = NameConverter(this.name);
        rule.parent_node = my_name;
        rule.parent_position = gameObject.transform.position.ToString();
    }

    private void OnApplicationQuit()
    {
        rule.SortLists();
        rule.parent_position = Vector3.zero.ToString();
        string json_content = Newtonsoft.Json.JsonConvert.SerializeObject(rule);

        string my_name = NameConverter(this.name);
        string file_name = my_name + "_.json";
        int index = 0;

        while (File.Exists(output_path + file_name.Replace("_.", "_" + index.ToString() + ".")))
        {
            index += 1;
        }
        string path = output_path + file_name.Replace("_.", "_" + index.ToString() + ".");

        FileStream fileStream = new FileStream(path, FileMode.Create);
        byte[] data = Encoding.UTF8.GetBytes(json_content);
        fileStream.Write(data, 0, data.Length);
        fileStream.Close();
    }

    private void OnTriggerEnter(Collider other)
    {
        string my_name = NameConverter(this.name);
        string other_name = NameConverter(other.name);


        Vector3 direction = other.gameObject.transform.position - this.gameObject.transform.position;
        direction = direction / direction.magnitude;


        rule.child_nodes.Add(other_name);
        rule.child_directions.Add(direction.ToString());
        rule.child_positions.Add((other.transform.position - transform.position).ToString());
    }

    private string NameConverter(string name)
    {
        if (name.IndexOf(" (") != -1)
            name = name.Substring(0, name.IndexOf(" ("));

        return name;
    }
}
