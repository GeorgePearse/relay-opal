# relay-opal

Relay, a forward-only credit-assignment estimator for pre-norm transformers, packaged as a PyTorch library, with
OPAL's coding agent allowed to rewrite the estimator and the optimizer while the model trains.

Work in progress: the skeleton is here, the estimator, the fidelity check, the OPAL hybrid and the Track 1 example
follow in the next pull requests. See CHANGELOG.md.

## Credits

- Relay: Emanuele Rodolà (@erodola), qlabs-eng/noise PR #20.
- CutBI and hidden-unit reassociation: @Euraba, qlabs-eng/noise PRs #17 and #15.
- Boost -> Cut, the base script: @spicylemonade, qlabs-eng/noise PR #11.
- OPAL: Xingjian Bai and co-authors, https://github.com/xingjian-bai/opal.

## Licence

MIT. Derived code keeps its upstream notices in THIRD_PARTY_NOTICES.md.
