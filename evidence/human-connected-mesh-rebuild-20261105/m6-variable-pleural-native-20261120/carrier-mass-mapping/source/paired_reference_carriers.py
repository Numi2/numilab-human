"""Rebase the archived carrier barycentrics to the current registered reference."""
import hashlib
from pathlib import Path
import sys

import numpy as np


def paired_reference_carriers(compiler, source_nodes, source_tets):
    previous = Path('../resting-clearance-and-atrium-010')
    sys.path.insert(0, str(previous.resolve()))
    from rebind_current_skin import read_literal

    paired = previous / 'paired-skin-fat-refit-001/paired-geometry-and-mass.npz'
    registration = previous / 'paired-registration-with-face-ancestry/registration.npz'
    inputs = {
        paired: '562256073b6c6b97292208201ecb0c868516b0fda7b5bd80479f3e45e450dec0',
        registration: 'dfbff6e8891f5d04059c6a44858213c079b16ec93a552049171a41feeaba5d39',
    }
    for path, expected in inputs.items():
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'reference hash mismatch: {path}')

    source = np.loadtxt(source_nodes, skiprows=1, delimiter=',')
    topology = np.loadtxt(source_tets, skiprows=1, delimiter=',', dtype=int, usecols=range(7))
    tet = topology[:, 3:7]
    assert np.array_equal(source[:, 0], np.arange(len(source)))
    assert np.array_equal(topology[:, 0], np.arange(len(tet)))
    with np.load(paired, allow_pickle=False) as packet:
        points = packet['points_world_m'].reshape(-1, 3)
        material = packet['child_material']
        assert np.array_equal(tet, packet['child_tetrahedra'])
        assert np.array_equal(material + 14, topology[:, 1])
        element = points[tet]
        edge = element[:, 1:] - element[:, :1]
        volume = np.einsum('ij,ij->i', edge[:, 0], np.cross(edge[:, 1], edge[:, 2])) / 6.
        mass = volume * np.take([0., 1000., 900.], material)
        assert np.all(mass > 0.)
        nodal = np.bincount(tet.ravel(), weights=np.repeat(mass * .25, 4), minlength=len(points))
        error = float(np.abs(nodal - packet['node_mass_kg']).max())
        assert error <= 1e-14

    owner = source[:, 9].astype(int)
    with np.load(registration, allow_pickle=False) as packet:
        assert np.array_equal(np.tile(packet['donor_body'], 3), owner)
        reference_to_donor = np.tile(packet['world_to_donor'].reshape(-1, 3, 3), (3, 1, 1))
    registered_frames = np.loadtxt(compiler/'volume-export/registered/donor-frames.csv',
                                   skiprows=1, delimiter=',')
    by_body = {int(row[1]): row for row in registered_frames}
    frames = np.asarray([by_body[body] for body in owner])
    donor_to_assembly = frames[:, 5:14].reshape(-1, 3, 3)
    current = source[:, 2:5]
    observed = source[:, 10:13]
    transforms = (donor_to_assembly, np.swapaxes(donor_to_assembly, 1, 2))
    errors = [np.linalg.norm(frames[:, 2:5] + np.einsum('nij,nj->ni', transform, observed)
                             - current, axis=1).max() for transform in transforms]
    selected = int(np.argmin(errors))
    assert errors[selected] < 3e-6
    apps = Path.home()/'Work/skin-fat-native-integration-209/apps'
    coarse = apps/'NumiHumanWholeBodySkinReduction.hpp'
    candidate = apps/'NumiHumanFinal128CandidateRegistration.hpp'
    read = lambda path, key: read_literal(path, key).reshape(-1, 3)
    historical = np.vstack((
        read(candidate, 'kWholeBodySkinCandidateReferenceWorldPoints')
        + read(coarse, 'kWholeBodySkinMechanicalWorldDeltas'),
        read(coarse, 'kWholeBodySkinInnerWorld'),
        read(coarse, 'kWholeBodyFatDeepWorld'),
    ))
    assert historical.shape == points.shape == current.shape
    offset = np.einsum('nij,njk,nk->ni', transforms[selected], reference_to_donor, points-historical)
    report = {
        'budget': 'current_paired_reference',
        'inputs_sha256': {str(path): sha for path, sha in inputs.items()},
        'same_ordered_tetrahedra_and_materials': len(tet),
        'current_nodal_reproduction_max_error_kg': error,
        'donor_frame_reconstruction_max_error_m': float(errors[selected]),
        'maximum_reference_carrier_frame_change_m': float(np.linalg.norm(offset, axis=1).max()),
    }
    return source[:, 5:8] + offset, tet, mass, report
