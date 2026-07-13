using ProceduralConstructionPitGenerator;
using UnityEditor;
using UnityEngine;

namespace ProceduralConstructionPitGenerator.Editor
{
    [CustomEditor(typeof(ProceduralConstructionPit))]
    public class ProceduralConstructionPitEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            GUILayout.Space(10f);
            ProceduralConstructionPit pit = (ProceduralConstructionPit)target;

            using (new GUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Generate / Rebuild"))
                {
                    Undo.RegisterFullObjectHierarchyUndo(pit.gameObject, "Generate Construction Pit Hazard");
                    pit.Generate();
                    EditorUtility.SetDirty(pit.gameObject);
                }

                if (GUILayout.Button("Clear Generated"))
                {
                    Undo.RegisterFullObjectHierarchyUndo(pit.gameObject, "Clear Construction Pit Hazard");
                    pit.ClearGenerated();
                    EditorUtility.SetDirty(pit.gameObject);
                }
            }
        }

        [MenuItem("GameObject/3D Object/Construction Pit Hazard", false, 18)]
        private static void CreateConstructionPit(MenuCommand command)
        {
            GameObject go = new GameObject("Construction Pit Hazard");
            ProceduralConstructionPit pit = go.AddComponent<ProceduralConstructionPit>();
            pit.Generate();

            GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            Undo.RegisterCreatedObjectUndo(go, "Create Construction Pit Hazard");
            Selection.activeObject = go;
        }
    }
}
