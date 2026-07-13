using ProceduralManholeGenerator;
using UnityEditor;
using UnityEngine;

namespace ProceduralManholeGenerator.Editor
{
    [CustomEditor(typeof(OpenManholeHazardScenario))]
    public class OpenManholeHazardScenarioEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            GUILayout.Space(10f);
            OpenManholeHazardScenario scenario = (OpenManholeHazardScenario)target;

            using (new GUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Generate Scenario"))
                {
                    Undo.RegisterFullObjectHierarchyUndo(scenario.gameObject, "Generate Open Manhole Hazard Scenario");
                    scenario.GenerateScenario();
                    EditorUtility.SetDirty(scenario.gameObject);
                }

                if (GUILayout.Button("Apply Cover State"))
                {
                    Undo.RegisterFullObjectHierarchyUndo(scenario.gameObject, "Apply Open Manhole Cover State");
                    scenario.ApplyCoverState();
                    EditorUtility.SetDirty(scenario.gameObject);
                }
            }
        }

        [MenuItem("GameObject/3D Object/Open Manhole Hazard Scenario", false, 19)]
        private static void CreateOpenManholeHazardScenario(MenuCommand command)
        {
            GameObject go = new GameObject("Open Manhole Hazard Scenario");
            OpenManholeHazardScenario scenario = go.AddComponent<OpenManholeHazardScenario>();
            scenario.GenerateScenario();

            GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            Undo.RegisterCreatedObjectUndo(go, "Create Open Manhole Hazard Scenario");
            Selection.activeObject = go;
        }
    }
}
