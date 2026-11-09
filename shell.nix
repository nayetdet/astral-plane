{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    glib
    expat
    python3
    uv
    libx11
    libxext
    libxfixes
    libxi
    libxinerama
    libxrandr
    libxrender
    libxcursor
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.glib
    pkgs.expat
    pkgs.libx11
    pkgs.libxext
    pkgs.libxfixes
    pkgs.libxi
    pkgs.libxinerama
    pkgs.libxrandr
    pkgs.libxrender
    pkgs.libxcursor
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ];
}
