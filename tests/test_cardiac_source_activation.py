"""Analytic continuous-travel and interface tests for the heart candidate."""

import subprocess
from pathlib import Path

import numpy as np

from numilab_human.cardiac_source_activation import _unique_faces


ROOT = Path(__file__).resolve().parents[1]


def test_complete_face_join_does_not_promote_point_contact():
    left = np.array([[0, 1, 2, 3]], dtype='<u4')
    right = np.array([[0, 1, 2, 4], [3, 5, 6, 7]], dtype='<u4')
    shared_faces = np.intersect1d(_unique_faces(left), _unique_faces(right))
    assert np.array_equal(shared_faces.view('<u4').reshape(-1, 3), [[0, 1, 2]])
    face_nodes = np.unique(shared_faces.view('<u4'))
    shared_nodes = np.intersect1d(np.unique(left), np.unique(right))
    assert np.array_equal(np.setdiff1d(shared_nodes, face_nodes), [3])


def test_tetrahedral_update_takes_continuous_face_path(tmp_path):
    source = tmp_path/'analytic.cpp'
    executable = tmp_path/'analytic'
    source.write_text('''
#define main cardiac_refine_main
#include "''' + str(ROOT/'tools/cardiac_activation_refine.cpp') + '''"
#undef main
int main() {
    // A zero-time triangular face at z=0, target above its interior.
    // Edge-only Dijkstra gives >1; the continuous face update gives 1.
    double t[4]={1e9,0,0,0};
    Vec p[4]={{.3,.3,1},{0,0,0},{1,0,0},{0,1,0}};
    double d[4][4]{};
    for(int i=0;i<4;++i)for(int j=0;j<4;++j){
        Vec delta=sub(p[i],p[j]);d[i][j]=dot(delta,delta);
    }
    double arrival=candidate(t,d,0);
    return std::abs(arrival-1.0)<1e-12 ? 0 : 3;
}
''')
    subprocess.run(['clang++', '-std=c++20', '-O2', str(source), '-o',
                    str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
