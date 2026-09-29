# Z-Anatomy thorax geometry attribution

This selected geometry export and its registered lung-lobe/pleural geometry
derivatives are distributed under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Credits required by the pinned upstream source:

- BodyParts3D - The Database Center for Life Science - CC-BY-SA 2.1 Japan
- Z-Anatomy - The libre 3D atlas of anatomy - CC-BY-SA 4.0

Original BodyParts3D model: Kousaku Okubo. Z-Anatomy design, 3D and anatomy:
Gauthier Kervyn. Blender add-on: Marcin Zielinski. Unity development: Lluis Vinent.
The add-on and other embedded source scripts were not executed.

Source repository and license/attribution:
[pinned Z-Anatomy README](https://github.com/Z-Anatomy/Models-of-human-anatomy/blob/9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a/Readme.md).

Revision: `9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a`.
Archive: `Z-Anatomy.zip`, SHA-256
`e029688545627bd0214b269e1063143abb580aad72b2c2445d6d8a9a0d9da736`.
Selected member: `Z-Anatomy/Startup.blend`, SHA-256
`9f08a17ea0115fed80b2a73ecdf0a1bc2ab2f6956f37c593ce23d513ea35afcd`.

Numi adaptation: Blender 5.1.2 evaluated viewport geometry was exported for
the five authored lung-lobe mesh objects, the authored Pleura object and
27 thorax bone anchors. Existing source modifiers were retained. Polygons
were triangulated with Blender's `calc_loop_triangles`; area-weighted vertex
normals were calculated. No additional modifiers, repairs, decimation or
vertex fusion were applied. Selected surfaces were aligned with a proper
uniform similarity fitted to 13 bone surface centroids; 14 other bone
centroids remained held out. Native derivatives use registered metre units
and the MyoSim torso COM frame. No clinical validation is implied.

The whole upstream blend is retained locally for reproduction and is not
redistributed here. This selection contains no upstream definitions,
embedded add-on, kidney, inner-ear, brain or other organ meshes.

The separate 304-surface BodyParts3D baseline retains its existing source
attribution. The composite's geometry and rendered derivatives retain these
Z-Anatomy share-alike credits in addition to the baseline credits.
