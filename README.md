# VQ-VAE

The VQ-VAE implementation and its commit history were extracted from [another repo](https://github.com/ivannz/nle_toolbox.git) into this standalone repo.

## Extracting VQVAE from nle-toolbox


```bash
brew install git-filter-repo

# rm -rf nle_toolbox--delete
git clone https://github.com/ivannz/nle_toolbox.git nle_toolbox--delete
cd nle_toolbox--delete

# remove each file other than VQ
git filter-repo --invert-paths --force --path nle_toolbox/zoo/__init__.py
git filter-repo --invert-paths --force --path nle_toolbox/zoo/blstats.py
git filter-repo --invert-paths --force --path nle_toolbox/zoo/glyph.py
git filter-repo --invert-paths --force --path nle_toolbox/zoo/models
git filter-repo --invert-paths --force --path nle_toolbox/zoo/transformer
git filter-repo --invert-paths --force --path nle_toolbox/zoo/legacy

# 
git filter-repo --path nle_toolbox/zoo/ --path-rename 'nle_toolbox/zoo/':''

# create an export bundle
git bundle create vqvae.bundle --all
```
