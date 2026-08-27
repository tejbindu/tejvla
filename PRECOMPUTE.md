- Command for running:
```
lerobot-train \
            --policy.type babyvla \
            --dataset.repo_id lerobot/svla_so101_pickplace \
            --steps 200 \
            --policy.repo_id tejbindu/babyvla
```

```
lerobot-train --config_path policies/lerobot_policy_babyvla/config/train.json
```

- Setup:
```
We must downgrade ffmpeg
Install: sudo pacman -S libglvnd ffmpeg4.4 
```

#### Notes
- Tried pusht dataset but didnt work. too many issues. so moved on and tried libero dataset. That had issues with dataset size. so moving to the lerobot svla pickplace dataset
