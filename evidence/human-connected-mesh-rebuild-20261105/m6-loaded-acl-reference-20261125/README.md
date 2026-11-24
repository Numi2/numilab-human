# Supported regional motion with reference-displacement anterior cruciate ligaments

The four retained failures in acl-witnesses.json all put the peak FEM momentum residual on interior node 7, 8 or 9 of the first 16-node ACL. The reporting continuum object ID describes the object that first published the shared certificate. The packed residual tag identifies the peak row independently.

Both registered ACLs now use the same material reference displacement formulation as the fascia. The compiler checks the emitted source nodes, topology, zero initial displacement and all 13 material values per side. The attached simulation uses the fixed authored parameters. The identification priors remain in the source material. The change retains the original constitutive law, density, connectivity, entheses, load rows and reaction owners. The native compilation gate rejected the two intermediate attempts with active identification distributions. The recovery archive includes these trials.

| Same loaded pose, muscle and bed owners | Previous ACL formulation | Reference-displacement ACLs |
| --- | ---: | ---: |
| Accepted steps of eight at 0.125 ms | 1 | 8 |
| First step GPU time | 2,215 ms | 689 ms |
| First step active Krylov columns | 1,024 | 128 |
| First step normal bed reaction | 960.948 N | 960.948 N |
| Total modeled mass difference between configurations |  | less than 1 nanogram |

A separate 64-step run completed 8.000 ms with the registered 34 ligament force rows, 26 fascia regions, regional skin/fat bed, variable pleural pressure, cardiac, gas and Brain owners. It took 46.46 s in the native horizon and 61.49 s from launch through exit. Measured throughput was 0.000172× in the horizon and 0.000130× end to end. At step 64 the normal reaction was 944.91 N, tangential reaction was 216.17 N and the center of mass was moving sideways at 20.75 mm/s. Static equilibrium remains open.

At 2 ms the previous ACL formulation rejected the first root. The reference-displacement formulation accepted one root and rejected the next on vascular row 19 at 1.175e-5 against 1e-5. The matched vascular block/cavity preconditioner also accepted one root and then rejected on the physical mechanical certificate. Neither trial was promoted. The full dense skin/fat model remains a separate qualification.

The exact skin audit found 11,667 skin-to-internal triangle pairs at 1 ms and 11,672 at 8 ms. Of these, 11,547 had identical anatomical bank and face identities across both captures. At 8 ms the pairs involved 2,856 bone triangle pairs, 8,189 muscle triangle pairs, and 627 vessel triangle pairs. Skin had three additional self-intersecting pairs at each capture. These counts are triangle pairs, without a between-frame guarantee. The GPU geometry packs, source hashes and complete individual witnesses are recoverable.

On SSH Numi the eight-millisecond run is

```sh
cd ~/Work/numi-human-supported-shoulder-085/force-route-086
./run-supported-scene --launch connected-forearms-087/acl-material-64-source/launch.json
```

The reviewable native source is under tools/native_ligament_contact/registered_material_volume/acl. Its emitter was tested against the previous pinned native source and reproduced the compiled source byte for byte. The exact binary, launch configurations, input pins and logs are in the independently downloaded, member-verified archive recorded in acl-recovery.json.
