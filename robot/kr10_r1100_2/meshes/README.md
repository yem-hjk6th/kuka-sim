# Meshes (not in this repository)

The URDF refers to `base_link.STL` and `link_1.STL` … `link_6.STL` in this folder. They are not committed, because the licence of the CAD data they were exported from has not been checked.

Code in this repository must run without them: use primitive collision shapes (capsules or boxes per link) when the files are missing. If you have the files locally, put them here; `.gitignore` keeps them out of git.
