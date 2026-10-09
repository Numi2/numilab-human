# Implemented scope and limits

This scene is a mixed-source reference adult supported by bed contact. It is not a measured reconstruction of one person. Source identities, laterality, registration, inferred geometry, and licenses remain in the retained anatomy manifests and [source credits](../native-resting-launchers-1165/ATTRIBUTION.md).

## Functional owners

- The existing Metal articulated-body runtime owns body dynamics and contact. In the executed viewer, registered NHSKIN vertices and weights also determine per-region support minima and weighted point Jacobians for the existing NHCNT queries; these are discrete point contacts, not triangle-mesh collision. The older NHSKIN manifest's visual-only wording does not describe this runtime use. Skin changes must therefore be checked for contact effects. The reference has 157 bodies, 96 with mass, 72 kg total physical mass, and 32 bed-contact regions. Organ and blood masses are not added a second time. The mass scale from the source-body aggregate is declared, rather than treating the source aggregate as a measured subject mass.
- Existing MyoSim muscle mechanics and compliant tendon state drive two respiratory modal coordinates. Diaphragm and thoracic displacement determine lung volume; compliance and resistance determine pressure and airflow. There is no direct respiratory modal reaction-force or modal mass-redistribution term in full-body articulated q/v. The skin-support query path is a separate contact coupling and must be inspected before claiming complete mechanical isolation.
- The GPU respiratory/gas model uses a 2.5 L reference FRC, 150 mL dead space, lung/chest compliance of 0.2 L/cmH2O each, and reference airway resistance of 133322 Pa s/m3. Tissue reference metabolism is 250 mL O2 STPD/min and 200 mL CO2 STPD/min. Parameters are declared reference assumptions.
- The existing 21-compartment, 24-edge CVSim circulation owns chamber and vascular volumes, pressure, valve flow, and blood accounting. Seven anatomical cardiac deformation coordinates bind calculated chamber function to the displayed heart. The anatomical leaflets are passive representations; this is not resolved 3D leaflet/fluid interaction, and cardiac deformation does not contribute reaction forces to articulated-body momentum.
- Pulmonary perfusion affects gas exchange, systemic metabolism changes blood gas content, and the existing Brain respiratory controller responds to that coupled gas state. Respiratory-pressure effects on cardiovascular hemodynamics are not established here. There is no demonstrated complete autonomic cardiovascular controller or direct force-balanced whole-body response to the respiratory modal coordinates.
- The native viewer uses accepted state for anatomical deformation and synchronized measurements. Physical and physiological updates remain in the existing Metal path; CPU work loads assets, schedules work, records observations, and presents frames.

## Anatomical interpretation

The 1159 lung repair changes eight source vertices and their eight exact pleural-proxy copies, by at most 2.002 micrometres, while preserving registered interface triangles. Detailed surfaces are inspection geometry, not millions of independently simulated tissue elements.

The skin is an inferred enclosing reference surface. The retained 927 construction moved 9513 of 54663 reference vertices, by at most 18.395 mm. Its ocular boundary loops remain open. Neither open-surface volume integrals nor mixed source organ volumes establish a closed whole-body density measurement.

Row 310 is an external lung-union/visceral proxy. It is not an explicit parietal pleura, pleural-fluid layer, or complete fissure lining. The liver exterior retains source identity while its retired overlapping alias is hidden; this does not provide internal hepatic segmentation.

The [right-heart shared-boundary certificate](../native-cardiac-shared-interface-1182/README.md) verifies disjoint right-atrial/right-ventricular interiors in current source-neutral anatomy, the registered baseline's terminal accepted state, and both arms' accepted captures at 108.094 s (after the 60–100 s intervention). Its 11,568 exact triangle contacts are confined to 690 opposite-winding shared triangles and their boundary features. This is an inferred geometric compartment interface; it does not establish biological valve-plane/orifice ownership, all cardiac tissue interfaces, or continuous-time clearance. Historical raw-source overlap counts do not describe this later registered geometry.

## Evidence boundaries

Full lung and skin audits evaluate the exact discrete accepted captures named in each report, with source-defined intended interfaces distinguished from unclassified intersections. They do not prove continuous-time clearance between captures. Short initial-cycle evidence and late captures have separate identities.

The physiological model omits detailed regional ventilation/perfusion heterogeneity, pH/Haldane effects, cellular biology, digestion, endocrine regulation, and complete physiology of passive organs. Reference-range comparisons indicate plausibility for particular observables, not clinical validity.

The retained rejection/replay checks establish unchanged tested body/Brain accepted state and a fresh-context six-root replay comparison. They do not establish complete same-context archive rollback/retry identity. The prior compliance/force/resistance sensitivity probes cover two post-initialization breaths with the same physical parameter family; they are not a long-horizon sweep of metabolic rates or controller gains.

The 018 viewer was built from a dirty source tree. Exact build source pins and the source-delta patch are retained; the later clean commit must not be mistaken for a clean-at-build assertion. The source revisions and binary identities are listed in provenance/source-revisions-final.json.

Current closed native inputs and results use reversible macOS user-immutable flags. A separate earlier cleanup deleted the raw 917 trial and some historical references. That incident and its failed-attempt records are retained; historical compact summaries do not replace the fresh 1170 measurements.
