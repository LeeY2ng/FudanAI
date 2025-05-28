import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class MYTrainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
    ):
        self.args = args

        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # 混合精度训练
        self.scaler = torch.amp.GradScaler(enabled=self.fp16)

        # 创建 DataLoader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )
        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.args.per_device_eval_batch_size
        )

    def train(self):
        # 训练循环
        global_step = 0
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

            for batch in progress_bar:
                # 数据移到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 混合精度上下文
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.args.fp16
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                # 反向传播
                self.scaler.scale(loss).backward()

                # 参数更新
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # 日志记录
                if global_step % self.args.logging_steps == 0:
                    progress_bar.set_postfix(loss=loss.item())

                # 评估
                if global_step % self.args.eval_steps == 0:
                    self.model.eval()
                    eval_loss = 0
                    for eval_batch in self.eval_dataloader:
                        eval_batch = {
                            k: v.to(self.device) for k, v in eval_batch.items()
                        }
                        with torch.no_grad():
                            outputs = self.model(**eval_batch)
                        eval_loss += outputs.loss.item()
                    eval_loss /= len(self.eval_dataloader)
                    print(f"Step {global_step}: Eval Loss = {eval_loss:.4f}")
                    self.model.train()

                # 模型保存
                if global_step % self.save_steps == 0:
                    save_path = os.path.join(
                        self.args.output_dir, f"checkpoint-{global_step}"
                    )
                    self.model.save_pretrained(save_path)

            global_step += 1

