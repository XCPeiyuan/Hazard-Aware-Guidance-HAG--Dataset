using ProceduralManholeGenerator;
using UnityEditor;
using UnityEngine;

namespace ProceduralManholeGenerator.Editor
{
    [CustomEditor(typeof(ProceduralManhole))]
    public class ProceduralManholeEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            GUILayout.Space(10f);
            ProceduralManhole manhole = (ProceduralManhole)target;

            using (new GUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Generate / Rebuild"))
                {
                    Undo.RegisterFullObjectHierarchyUndo(manhole.gameObject, "Generate Procedural Manhole");
                    manhole.Generate();
                    EditorUtility.SetDirty(manhole.gameObject);
                }

                if (GUILayout.Button("Clear Generated"))
                {
                    Undo.RegisterFullObjectHierarchyUndo(manhole.gameObject, "Clear Procedural Manhole");
                    manhole.ClearGenerated();
                    EditorUtility.SetDirty(manhole.gameObject);
                }
            }
        }

        [MenuItem("GameObject/3D Object/Procedural Manhole", false, 20)]
        private static void CreateProceduralManhole(MenuCommand command)
        {
            GameObject go = new GameObject("Procedural Manhole");
            ProceduralManhole manhole = go.AddComponent<ProceduralManhole>();
            manhole.Generate();

            GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            Undo.RegisterCreatedObjectUndo(go, "Create Procedural Manhole");
            Selection.activeObject = go;
        }
    }
}
