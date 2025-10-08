import torch

def PositionEmbedding(time_steps, temb_dim):
    # 割り方を変えた。エラーが出るかも。
    factor = 10000 ** (torch.arange(start=0, end=temb_dim//2, dtype=torch.float32, device=time_steps.device) * 2 / temb_dim)
    # まずはtime_stepsを縦ベクトルに変換。その後、それを横に並べる感じ。
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

