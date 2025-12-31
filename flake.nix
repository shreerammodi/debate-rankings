{
  description = "Python development environment for computing ratings";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShell =
          with pkgs;
          mkShell {
            packages = [
              black
              python313
            ];

            buildInputs = [
              python313
              python313Packages.pip
              python313Packages.virtualenv
            ];

            shellHook = ''
              # Create a virtual environment if it doesn't exist
              if [ ! -d .venv ]; then
                echo "Creating virtual environment..."
                python -m venv .venv
              fi

              # Activate the virtual environment
              source .venv/bin/activate

              # Install skelo if not already installed
              if ! python -c "import skelo" 2>/dev/null; then
                echo "Installing skelo..."
                pip install skelo
              fi
            '';
          };
      }
    );
}
