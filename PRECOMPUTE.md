- Command for running:
```
lerobot-train \
            --policy.type babyvla \
            --dataset.repo_id lerobot/svla_so101_pickplace \
            --steps 200000 \
            --policy.repo_id tejbindu/babyvla
```

- Setup:
```
Install: sudo pacman -S libglvnd
```

#### Notes
- Tried pusht dataset but didnt work. too many issues. so moved on and tried libero dataset
