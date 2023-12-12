import torch
import json

def train_model(train_loader, val_loader, generator, discriminator, criterion, optimizer_G, optimizer_D, device, num_epochs=50):
    record_dic = dict()
    for epoch in range(num_epochs):
        epoch_dic = dict()
        generator.train()
        discriminator.train()
        # 初始化损失统计
        running_loss_G = 0.0
        running_loss_D = 0.0

        # 训练阶段
        for i, (images, targets, texts_embeddings) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            texts_embeddings = texts_embeddings.to(device)
            # Train Discriminator
            optimizer_D.zero_grad()

            # Real samples
            real_labels = torch.ones(images.size(0), 1).to(device)    
            outputs = discriminator(targets)
            d_loss_real = criterion(outputs, real_labels)

            # Fake samples
            fake_targets = generator(images, texts_embeddings)
            fake_labels = torch.zeros(images.size(0), 1).to(device)
            outputs = discriminator(fake_targets.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            # 判别器的总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_targets)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

            # 更新运行损失
            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()

            # 每处理完一定数量的批次后打印信息
            if (i + 1) % 10 == 0:  # 假设每100个批次打印一次
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Generator Loss: {running_loss_G / (i + 1)}, Discriminator Loss: {running_loss_D / (i + 1)}')
            step_dic = dict()
            step_dic['G LOSS'] = running_loss_G / (i + 1)
            step_dic['D LOSS'] = running_loss_D / (i + 1)
            epoch_dic[f"Step [{i + 1}/{len(train_loader)}]"] = step_dic

        # 验证阶段
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, targets, texts_embeddings in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                texts_embeddings = texts_embeddings.to(device)
                # 只计算生成器的损失
                fake_targets = generator(images, texts_embeddings)
                outputs = discriminator(fake_targets)
                val_loss += criterion(outputs, torch.ones(images.size(0), 1).to(device)).item()

        # 打印损失信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: G - {g_loss.item()}, D - {d_loss.item()}, Val Loss: {val_loss / len(val_loader)}')


        step_dic = dict()
        step_dic["Train G Loss"] = g_loss.item()
        step_dic["Train D Loss"] = d_loss.item()
        step_dic["Val Loss"] = val_loss / len(val_loader)
        epoch_dic["Final"] = step_dic
        record_dic[epoch] = epoch_dic
    with open('loss_total.json', 'w') as f:
        json.dump(record_dic, f)
