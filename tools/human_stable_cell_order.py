"""Exact cell permutation inside the existing replacement/ancestry protocol."""
import numpy as np

def replacement_in_reference_order(cells,offsets,parents,reference):
 cells=np.asarray(cells,dtype=np.int64);reference=np.asarray(reference,dtype=np.int64)
 offsets=np.asarray(offsets,dtype=np.int64);parents=np.asarray(parents,dtype=np.int64)
 if cells.shape!=reference.shape or cells.ndim!=2 or cells.shape[1]!=4:raise ValueError('stable cell topology domains differ')
 if offsets.shape!=(len(cells)+1,) or offsets[0]!=0 or offsets[-1]!=len(parents) or np.any(np.diff(offsets)<=0):raise ValueError('invalid stable cell ancestry CSR')
 def keys(t):return np.ascontiguousarray(np.sort(t,axis=1)).view(np.dtype((np.void,32))).reshape(-1)
 k=keys(cells);order=np.argsort(k);ordered=k[order];wanted=keys(reference)
 if np.any(ordered[1:]==ordered[:-1]) or len(np.unique(wanted))!=len(wanted):raise ValueError('duplicate stable cell identity')
 hits=np.searchsorted(ordered,wanted)
 if np.any(hits>=len(ordered)) or not np.array_equal(ordered[hits],wanted):raise ValueError('stable reference changes a source cell')
 permutation=order[hits];counts=np.diff(offsets)[permutation];new_offsets=np.r_[0,np.cumsum(counts)]
 indices=np.repeat(offsets[permutation]-new_offsets[:-1],counts)+np.arange(len(parents))
 return reference.copy(),new_offsets,parents[indices],permutation
