# FAQ

## Installation

### How do I download `curl`?
For smcore 0.1.0, it requires curl (usually called within `pycurl` in Python).

#### Seeing if `curl`/`pycurl` works:
If you want to check if it works, run the following in a Python interactive shell:
```python
import pycurl
```

You can also try:
```bash
### https://stackoverflow.com/questions/13449054/check-to-see-if-curl-is-installed-locally

### for mac/linux
curl -V

### or for windows
curl.exe -V 
```

#### To install
Within Linux, run the following:

```
sudo apt-get update
sudo apt install libcurl4-openssl-dev libssl-dev
```

With a Mac, it may be a little more complicated.

For an M1 Mac:
```bash
### within a Terminal window ###

# in case pycurl was already installed
pip uninstall pycurl 

# install curl on your computer
# requires homebrew to be installed on your computer. if it hasn't been (https://stackoverflow.com/questions/21577968/how-to-tell-if-homebrew-is-installed-on-mac-os-x), please see below.
brew install curl-openssl

export PYCURL_CURL_DIR=$(brew --prefix curl-openssl)
export PYCURL_CURL_CONFIG=$PYCURL_CURL_DIR/bin/curl-config

### optional? ###
brew install openssl

OPENSSL_DIR=$(brew --prefix openssl) \
LDFLAGS="-L${OPENSSL_DIR}/lib" CPPFLAGS="-I${OPENSSL_DIR}/include" \
PKG_CONFIG_PATH="${OPENSSL_DIR}/lib/pkgconfig" \
##############

# install pycurl
PYCURL_SSL_LIBRARY=openssl LDFLAGS=$LDFLAGS CPPFLAGS=$CPPFLAGS PKG_CONFIG_PATH=$PKG_CONFIG_PATH pip3 install --no-cache-dir pycurl
```

If this doesn't work for your Mac, please consult the following StackOverflow references in case they contain useful information for you:
- https://stackoverflow.com/questions/47888757/importerror-pycurl-libcurl-link-time-ssl-backend-openssl-is-different-from-c
- https://stackoverflow.com/questions/72884308/error-openssl-crypto-h-file-not-found-when-installing-pycurl-on-mac-using-pip
- https://superuser.com/questions/1492759/upgrading-mac-from-high-sierra-to-catalina-broke-import-pycurl


### How do I install `homebrew` for my M1+ Mac?


```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# https://stackoverflow.com/questions/66666134/how-to-install-homebrew-on-m1-mac

```

If you run into an error, could be because git checkout is limited. See this: https://stackoverflow.com/questions/70303947/error-installing-homebrew-unexpected-disconnect-while-reading-sideband-packet

To possibly remedy, run the following:

```bash
git config --global http.postBuffer 1048576000
```

If it still fails, it could be because of your internet connection (I had trouble running on the one of the weaker WIFI connections at UCLA, but it was fine once I went home).

After, there will be instructions at the bottom of your terminal output in case you need to add homebrew to your path. Something like this (but specific for your computer):

```bash
### Run these two commands in your terminal to add Homebrew to your **PATH**:

(echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/USERNAME/.zprofile

eval "$(/opt/homebrew/bin/brew shellenv)"
```
