{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    glib
    python3
    uv
    xorg.libX11
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.glib
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ];
}
