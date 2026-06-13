{
  description = "Readmission Prediction API Development Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      pythonPackages = pkgs.python312Packages;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          pythonPackages.pip
          pythonPackages.virtualenv
          stdenv.cc.cc.lib
          zlib
          libgcc
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
          
          if [ ! -d ".venv" ]; then
            echo "Creating virtual environment..."
            python -m venv .venv
          fi
          
          source .venv/bin/activate
          
          echo "Installing dependencies..."
          pip install -r requirements.txt --quiet
          
          alias start-server="uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload"
          
          echo "--------------------------------------------------"
          echo "🚀 Readmission API Dev Environment"
          echo "Python: $(python --version)"
          echo "Commands:"
          echo "  start-server - Run the FastAPI server"
          echo "--------------------------------------------------"
        '';
      };
    };
}