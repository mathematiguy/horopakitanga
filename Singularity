################# Header: Define the base system you want to use ################
# Reference of the kind of base you want to use (e.g., docker, debootstrap, shub).
Bootstrap: docker
# Select the docker image you want to use (Here we choose tensorflow)
From: texlive/texlive:latest-full-doc

# Environment variables that should be sourced at runtime.
%environment
    # use bash as default shell
    SHELL=/bin/bash
    PYTHON_VERSION=3.11
    PATH="/pkg/code:$PATH" # ensure julia bin is in the path
    export SHELL PYTHON_VERSION PATH
    export JULIA_DEPOT_PATH="/tmp/:${JULIA_DEPOT_PATH}"

%files
    requirements.txt requirements.txt
    ataarangi /pkg/ataarangi
    setup.py /pkg

%post
    # Create the /code directory inside the container
    mkdir -p /code

    echo "Setting environment variables"
    export DEBIAN_FRONTEND=noninteractive

    echo "Installing Tools with apt-get"
    apt-get update
    apt-get install -y curl \
            wget \
            unzip \
            software-properties-common \
            build-essential \
            rsync \
            git \
            entr \
            python3-launchpadlib
    apt-get clean

    # Install apt packages
    apt update
    add-apt-repository ppa:rmescandon/yq -y
    add-apt-repository ppa:deadsnakes/ppa -y

    # Install yq (yaml processing)
    apt install -y yq

    # Download and install URW Garamond font
    wget https://mirrors.ctan.org/fonts/urw/garamond.zip
    unzip garamond.zip
    mkdir -p /usr/local/texlive/texmf-local/fonts/type1/urw/garamond
    mkdir -p /usr/local/texlive/texmf-local/fonts/afm/urw/garamond
    cp garamond/ugmr8a.pfb garamond/ugmm8a.pfb garamond/ugmri8a.pfb garamond/ugmmi8a.pfb /usr/local/texlive/texmf-local/fonts/type1/urw/garamond/
    cp garamond/ugmr8a.afm garamond/ugmm8a.afm garamond/ugmri8a.afm garamond/ugmmi8a.afm /usr/local/texlive/texmf-local/fonts/afm/urw/garamond/

    # Unzip TeX support files
    unzip garamond/ugm.zip -d /usr/local/texlive/texmf-local/

    # Update filename database and font maps
    mktexlsr
    updmap-sys --enable Map=ugm.map    

    # Set the Python version
    PYTHON_VERSION=3.11
    export PYTHON_VERSION

    echo "Install python${PYTHON_VERSION}.."
    apt install -y python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils python${PYTHON_VERSION}-venv
    python${PYTHON_VERSION} -m venv /opt/local
    update-alternatives --install /usr/bin/python python /opt/local/bin/python${PYTHON_VERSION} 1
    update-alternatives --install /usr/bin/python3 python3 /opt/local/bin/python${PYTHON_VERSION} 1

    echo "Installing pip.."
    curl -sS https://bootstrap.pypa.io/get-pip.py | /opt/local/bin/python3
    update-alternatives --install /usr/local/bin/pip pip /opt/local/bin/pip${PYTHON_VERSION} 1
    update-alternatives --install /usr/local/bin/pip3 pip3 /opt/local/bin/pip${PYTHON_VERSION} 1

    echo "Install basic requirements"
    pip3 install --upgrade pip
    pip3 install wheel pyct

    echo "Installing python requirements"
    pip3 install -r requirements.txt
    pip3 install -e /pkg
