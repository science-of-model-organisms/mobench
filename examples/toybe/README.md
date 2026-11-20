# toybe

TOY. Not a behaviour we study. It exists to test the pipeline: `mobench fake` copies these files
into a scratch copy of the repo, at the paths a real behaviour uses, and builds an organism from
them with no GPU and no key. It is kept out of `data/` and `configs/` so that it can never be
counted, discovered or evaluated as a real behaviour.
