import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

class BatchRepresentation(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64)
        )
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.unsqueeze(-1).unsqueeze(-1)
        
class SelectGate(nn.Module):
    def __init__(self, channels, reduction=16, sparsity_target=0.5):
        super().__init__()
        self.channels = channels
        self.sparsity_target = sparsity_target

        self.register_buffer('static_importance_history', torch.ones(channels))
        self.register_buffer('dynamic_importance_history', torch.ones(channels))
        self.register_buffer('correlation_history', torch.ones(channels))
        self.register_buffer('feature_importance_history', torch.ones(channels))
        
        self.channel_importance = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.channel_importance, mean=0.0, std=0.01)
        
        self.feat_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + channels, channels),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            nn.Linear(channels, channels)
        )
        
        self.register_buffer('running_gates', torch.ones(channels) * 0.7)
        self.register_buffer('best_gates', torch.ones(channels))
        self.register_buffer('importance_score', torch.zeros(channels))
        self.register_buffer('gate_history', torch.zeros(100, channels))
        self.register_buffer('current_sparsity_loss', torch.tensor(0.0))
        
        self.history_ptr = 0
        self.momentum = 0.1
        self.training_phase = 'main_network'
        
        self.importance_weights = {
            'static': 0.3,
            'dynamic': 0.3,
            'feature': 0.2,
            'correlation': 0.2
        }
        
        self.loss_weights = {
            'sparsity': 1.0,
            'diversity': 0.1,
            'correlation': 0.1,
            'stability': 0.1,
            'consistency': 0.1
        }

    def calculate_losses(self, gates, features):
        losses = {}
        
        losses['sparsity'] = torch.abs(gates.mean() - self.sparsity_target)
        
        entropy = -torch.mean(gates * torch.log(gates + 1e-10) +
                            (1-gates) * torch.log(1-gates + 1e-10))
        losses['diversity'] = -entropy
        
        if features is not None:
            feat_flat = features.view(features.size(0), -1)
            corr_matrix = torch.matmul(feat_flat.T, feat_flat)
            corr_matrix = corr_matrix / (torch.norm(feat_flat, dim=0).unsqueeze(0) + 1e-10)
            losses['correlation'] = torch.mean(torch.abs(corr_matrix - torch.eye(
                corr_matrix.size(0), device=corr_matrix.device)))
        else:
            losses['correlation'] = torch.tensor(0.0, device=gates.device)
        
        losses['stability'] = F.mse_loss(gates.mean(0), self.running_gates)
        
        if self.history_ptr > 0:
            history_gates = self.gate_history[:self.history_ptr]
            mean_history = history_gates.mean(0)
            losses['consistency'] = F.mse_loss(gates.mean(0), mean_history)
        else:
            losses['consistency'] = torch.tensor(0.0, device=gates.device)
        
        total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())
        
        return total_loss, {k: v.item() for k, v in losses.items()}

    def update_importance_scores(self, x, gates):
        with torch.no_grad():
            static_imp = torch.sigmoid(self.channel_importance)
            self.static_importance_history.mul_(0.9).add_(static_imp.data * 0.1)
            
            dynamic_imp = x.mean([0, 2, 3]).abs()
            self.dynamic_importance_history.mul_(0.9).add_(dynamic_imp.data * 0.1)
            
            feat = self.feat_net(x).squeeze(-1).squeeze(-1)
            feature_imp = torch.sigmoid(feat.mean(0))
            self.feature_importance_history.mul_(0.9).add_(feature_imp.data * 0.1)
            
            b, c, h, w = x.size()
            x_flat = x.view(b, c, -1)
            corr_matrix = torch.matmul(x_flat, x_flat.transpose(1, 2))
            corr_matrix = corr_matrix / (torch.norm(x_flat, dim=2, keepdim=True) + 1e-10)
            correlation_penalty = 1 - torch.mean(torch.abs(corr_matrix), dim=1)
            self.correlation_history.mul_(0.9).add_(correlation_penalty.mean(0).data * 0.1)
    
    def get_combined_importance(self):
        static_imp = (self.static_importance_history - self.static_importance_history.min()) / \
                    (self.static_importance_history.max() - self.static_importance_history.min() + 1e-8)
        
        dynamic_imp = (self.dynamic_importance_history - self.dynamic_importance_history.min()) / \
                     (self.dynamic_importance_history.max() - self.dynamic_importance_history.min() + 1e-8)
        
        feature_imp = (self.feature_importance_history - self.feature_importance_history.min()) / \
                     (self.feature_importance_history.max() - self.feature_importance_history.min() + 1e-8)
        
        corr_imp = (self.correlation_history - self.correlation_history.min()) / \
                  (self.correlation_history.max() - self.correlation_history.min() + 1e-8)
        
        combined_importance = (
            self.importance_weights['static'] * static_imp +
            self.importance_weights['dynamic'] * dynamic_imp +
            self.importance_weights['feature'] * feature_imp +
            self.importance_weights['correlation'] * corr_imp
        )
        
        return combined_importance
        
    def forward(self, x, batch_repr):
        b, c, h, w = x.size()
        
        if self.training_phase == 'main_network':
            self.current_sparsity_loss = torch.tensor(0.0, device=x.device)
            return x
        
        feat = self.feat_net(x).squeeze(-1).squeeze(-1)  # [B, C]
        
        batch_feat = batch_repr.squeeze(-1).squeeze(-1)
        if batch_feat.dim() == 1:
            batch_feat = batch_feat.unsqueeze(0)
        
        combined_feat = torch.cat([
            batch_feat.expand(b, -1),
            feat
        ], dim=1)
        
        logits = self.fusion_net(combined_feat) + self.channel_importance
        
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            soft_gates = torch.sigmoid(logits + 0.1 * gumbel_noise)
            
            k = max(int(c * (1 - self.sparsity_target)), 1)
            sorted_gates, _ = torch.sort(soft_gates, dim=1, descending=True)
            threshold = sorted_gates[:, k-1:k]
            
            hard_gates = (soft_gates >= threshold).float()
            gates = hard_gates.detach() + soft_gates - soft_gates.detach()
            
            self.update_importance_scores(x, gates)
            
            with torch.no_grad():
                current_mean = gates.mean(0)
                self.running_gates = (1 - self.momentum) * self.running_gates + self.momentum * current_mean
                
                self.gate_history[self.history_ptr] = current_mean
                self.history_ptr = (self.history_ptr + 1) % self.gate_history.size(0)
                
                combined_importance = self.get_combined_importance()
                _, top_indices = torch.topk(combined_importance, k)
                self.best_gates.zero_()
                self.best_gates[top_indices] = 1.0
            
            self.current_sparsity_loss, _ = self.calculate_losses(gates, feat)
            
        else:
            gates = self.best_gates.expand(b, -1)
            self.current_sparsity_loss = torch.tensor(0.0, device=x.device)
        
        return x * gates.view(b, c, 1, 1)
    
    def set_training_phase(self, phase):
        self.training_phase = phase

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, sparsity_target=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.select_gate1 = SelectGate(out_channels, sparsity_target=sparsity_target)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.select_gate2 = SelectGate(out_channels, sparsity_target=sparsity_target)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.select_gate_shortcut = SelectGate(out_channels, 
                                                 sparsity_target=sparsity_target)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, batch_repr):
        identity = self.shortcut(x)
        if hasattr(self, 'select_gate_shortcut'):
            identity = self.select_gate_shortcut(identity, batch_repr)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.select_gate1(out, batch_repr)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.select_gate2(out, batch_repr)
        
        out += identity
        out = self.leaky_relu(out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, sparsity_target=0.5):
        super().__init__()
        
        self.batch_repr = BatchRepresentation(in_channels=3)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.select_gate1 = SelectGate(64, sparsity_target=sparsity_target)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.layer1 = self._make_layer(64, 64, 2, sparsity_target)
        self.layer2 = self._make_layer(64, 128, 2, sparsity_target, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, sparsity_target, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, sparsity_target, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, sparsity_target, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, sparsity_target))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 
                                   sparsity_target=sparsity_target))
        return nn.ModuleList(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_repr = self.batch_repr(x)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.select_gate1(x, batch_repr)
        x = self.relu(x)
        
        for block in self.layer1:
            x = block(x, batch_repr)
        for block in self.layer2:
            x = block(x, batch_repr)
        for block in self.layer3:
            x = block(x, batch_repr)
        for block in self.layer4:
            x = block(x, batch_repr)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def get_gates_status(self):
        gates_status = {}
        for name, module in self.named_modules():
            if isinstance(module, SelectGate):
                gates_status[name] = {
                    'running_gates': module.running_gates.clone(),
                    'best_gates': module.best_gates.clone(),
                    'importance_score': module.importance_score.clone(),
                    'active_ratio': module.best_gates.mean().item()
                }
        return gates_status

    def set_training_phase(self, phase):
        for module in self.modules():
            if isinstance(module, SelectGate):
                module.set_training_phase(phase)


class TwoStageTrainer:
    def __init__(self, model, train_loader, test_loader, device, 
                 learning_rate=0.001, weight_decay=5e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.initial_lr = learning_rate
        self.weight_decay = weight_decay
        
        self.metrics = {
            'stage1': {
                'train_loss': [], 'train_acc': [],
                'test_loss': [], 'test_acc': []
            },
            'stage2': {
                'train_loss': [], 'train_acc': [],
                'test_loss': [], 'test_acc': [],
                'gate_metrics': defaultdict(list)
            }
        }
        
        self.best_acc = {'stage1': 0, 'stage2': 0}
        self.patience = 15
        self.patience_counter = 0

    def setup_stage1(self):
        print("\nSetting up Stage 1 - Training ResNet backbone")
        
        self.model.set_training_phase('main_network')
        for name, param in self.model.named_parameters():
            if 'select_gate' in name or 'batch_repr' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.initial_lr,
            epochs=100,  
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3
        )


    def setup_stage2(self):
        
        print("\nSetting up Stage 2 - Joint training of SelectGate and ResNet")
        
        self.model.set_training_phase('gates')
        for param in self.model.parameters():
            param.requires_grad = True
        
        gate_params = []
        batch_repr_params = []
        network_params = []
        
        for name, param in self.model.named_parameters():
            if 'select_gate' in name:
                gate_params.append(param)
            elif 'batch_repr' in name:
                batch_repr_params.append(param)
            else:
                network_params.append(param)
        
        self.optimizers = {
            'gate': optim.AdamW(
                gate_params,
                lr=self.initial_lr,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            ),
            'batch_repr': optim.AdamW(
                batch_repr_params,
                lr=self.initial_lr * 0.5,
                weight_decay=0.01
            ),
            'network': optim.AdamW(
                network_params,
                lr=self.initial_lr * 0.1,
                weight_decay=self.weight_decay
            )
        }
        
        self.schedulers = {
            'gate': OneCycleLR(
                self.optimizers['gate'],
                max_lr=self.initial_lr,
                epochs=100,  
                steps_per_epoch=len(self.train_loader)
            ),
            'batch_repr': OneCycleLR(
                self.optimizers['batch_repr'],
                max_lr=self.initial_lr * 0.5,
                epochs=100,
                steps_per_epoch=len(self.train_loader)
            ),
            'network': OneCycleLR(
                self.optimizers['network'],
                max_lr=self.initial_lr * 0.1,
                epochs=100,
                steps_per_epoch=len(self.train_loader)
            )
        }

    def train_epoch_stage1(self, epoch, total_epochs):
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Stage 1 Epoch {epoch+1}/{total_epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total

    def train_epoch_stage2(self, epoch, total_epochs):
      
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        gate_metrics = defaultdict(float)
        
        pbar = tqdm(self.train_loader, desc=f'Stage 2 Epoch {epoch+1}/{total_epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            for opt in self.optimizers.values():
                opt.zero_grad()
            
            outputs = self.model(inputs)
            
            ce_loss = self.criterion(outputs, targets)
            gate_loss = torch.tensor(0.0, device=self.device)
            
            for name, module in self.model.named_modules():
                if isinstance(module, SelectGate):
                    gate_loss = gate_loss + module.current_sparsity_loss
            
            total_batch_loss = ce_loss + 0.1 * gate_loss
            
            total_batch_loss.backward()
            
            for param_group in self.model.parameters():
                torch.nn.utils.clip_grad_norm_(param_group, max_norm=1.0)
            
            for opt in self.optimizers.values():
                opt.step()
            
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            total_loss += total_batch_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            gate_metrics['ce_loss'] += ce_loss.item()
            gate_metrics['gate_loss'] += gate_loss.item()
            
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total, gate_metrics

    def evaluate(self, stage):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        self.metrics[stage]['test_loss'].append(avg_loss)
        self.metrics[stage]['test_acc'].append(accuracy)
        
        return avg_loss, accuracy

    def train_stage1(self, epochs=100):
        print("\nStarting Stage 1 Training - ResNet backbone")
        self.setup_stage1()
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch_stage1(epoch, epochs)
            self.metrics['stage1']['train_loss'].append(train_loss)
            self.metrics['stage1']['train_acc'].append(train_acc)
            
            test_loss, test_acc = self.evaluate('stage1')
            
            print(f"\nStage 1 Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")
            
            if test_acc > self.best_acc['stage1']:
                self.best_acc['stage1'] = test_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, test_acc, 'stage1')
                print(f"New best accuracy: {test_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print("\nEarly stopping triggered!")
                break
        
        return self.metrics['stage1']

    def train_stage2(self, epochs=100):
        print("\nStarting Stage 2 Training - Joint training")
        self.setup_stage2()
        

        for epoch in range(epochs):

            train_loss, train_acc, gate_metrics = self.train_epoch_stage2(epoch, epochs)
            self.metrics['stage2']['train_loss'].append(train_loss)
            self.metrics['stage2']['train_acc'].append(train_acc)
            
            test_loss, test_acc = self.evaluate('stage2')
            
            print(f"\nStage 2 Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

            
            if (epoch + 1) % 10 == 0:
                self.print_gates_status()
            
            if test_acc > self.best_acc['stage2']:
                self.best_acc['stage2'] = test_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, test_acc, 'stage2')
                print(f"New best accuracy: {test_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print("\nEarly stopping triggered!")
                break
        
        return self.metrics['stage2']

    def save_checkpoint(self, epoch, accuracy, stage):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics[stage],
            'best_acc': accuracy
        }
        
        if stage == 'stage1':
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            })
        else:
            checkpoint.update({
                'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
                'schedulers': {k: v.state_dict() for k, v in self.schedulers.items()}
            })
        
        torch.save(checkpoint, f'Resnet18_Cifar10_{stage}.pth')

    def print_gates_status(self):
        gates_status = self.model.get_gates_status()
        print("\nGates Status:")
        print("-" * 80)
        for name, status in gates_status.items():
            print(f"\n{name}:")
            print(f"Active ratio: {status['active_ratio']:.3f}")
            print(f"Mean importance: {status['importance_score'].mean():.3f}")
def print_shape_comparison(shape_comparison):
    print("\nLayer Shape Comparison (Original vs Pruned):")
    print("-" * 80)
    print(f"{'Layer Name':<30} {'Original Shape':<25} {'Pruned Shape':<25}")
    print("-" * 80)
    
    total_params_original = 0
    total_params_pruned = 0
    
    for name, shapes in shape_comparison.items():
        if 'original' in shapes and 'new' in shapes:
            orig_shape = shapes['original']['weight']
            new_shape = shapes['new']['weight']
            layer_type = shapes['original']['type']
            
            orig_params = np.prod(orig_shape)
            new_params = np.prod(new_shape)
            
            total_params_original += orig_params
            total_params_pruned += new_params
            
            reduction = (1 - new_params/orig_params) * 100
            
            print(f"{name:<30} {str(orig_shape):<25} {str(new_shape):<25}")
            print(f"{'Parameters:':<30} {orig_params:,} -> {new_params:,} ({reduction:.1f}% reduction)")
            print("-" * 80)
    
    total_reduction = (1 - total_params_pruned/total_params_original) * 100
    print(f"\nTotal Parameters:")
    print(f"Original: {total_params_original:,}")
    print(f"Pruned  : {total_params_pruned:,}")
    print(f"Overall reduction: {total_reduction:.1f}%")
    
def analyze_channel_correlation(model, loader, device):
    correlations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in correlations:
                correlations[name] = []
            activations = output.detach()
            b, c, h, w = activations.shape
            activations = activations.view(b, c, -1).mean(-1)
            correlations[name].append(activations)
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, SelectGate):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            if batch_idx >= 50:  
                break
            inputs = inputs.to(device)
            model(inputs)
    
    correlation_matrices = {}
    for name, activations_list in correlations.items():
        activations = torch.cat(activations_list, dim=0)
        correlation_matrix = torch.corrcoef(activations.t())
        correlation_matrices[name] = correlation_matrix
    
    for hook in hooks:
        hook.remove()
    
    return correlation_matrices

def select_channels_with_correlation(model, correlation_matrices, target_sparsity):
    selected_channels = {}
    
    for name, module in model.named_modules():
        if isinstance(module, SelectGate):
            static_importance = module.static_importance_history
            dynamic_importance = module.dynamic_importance_history
            correlation_penalty = module.correlation_history
            feature_importance = module.feature_importance_history
            
            static_importance = (static_importance - static_importance.min()) / (static_importance.max() - static_importance.min() + 1e-8)
            dynamic_importance = (dynamic_importance - dynamic_importance.min()) / (dynamic_importance.max() - dynamic_importance.min() + 1e-8)
            correlation_penalty = (correlation_penalty - correlation_penalty.min()) / (correlation_penalty.max() - correlation_penalty.min() + 1e-8)
            feature_importance = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min() + 1e-8)
            
            combined_importance = (
                0.25 * static_importance +   
                0.25 * dynamic_importance +  
                0.25  * feature_importance +  
                0.0 * correlation_penalty   
            )
            
            num_channels = len(module.running_gates)
            min_channels = max(1, int(num_channels * 0.1))
            num_keep = max(min_channels, int(num_channels * (1 - target_sparsity)))
            
            top_values, top_indices = torch.topk(combined_importance, num_keep)
            
            temperature = 2.0
            selected_probs = torch.zeros_like(combined_importance)
            selected_probs[top_indices] = F.softmax(top_values / temperature, dim=0)
            
            selected_channels[name] = np.sort(top_indices.cpu().numpy())
    
    return selected_channels

def record_shapes(model):
    shape_comparison = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            shape_comparison[name] = {
                'original': {
                    'weight': list(module.weight.detach().shape),
                    'type': module.__class__.__name__
                }
            }
    return shape_comparison

def create_pruned_model(model, selected_channels, shape_comparison):
    new_model = ResNet18(num_classes=10)
    
    with torch.no_grad():
        if hasattr(model, 'conv1'):
            first_layer_select = selected_channels.get('select_gate1', None)
            if first_layer_select is not None:
                new_model.conv1.weight.data = model.conv1.weight.data[first_layer_select]
                new_model.bn1.weight.data = model.bn1.weight.data[first_layer_select]
                new_model.bn1.bias.data = model.bn1.bias.data[first_layer_select]
                new_model.bn1.running_mean.data = model.bn1.running_mean.data[first_layer_select]
                new_model.bn1.running_var.data = model.bn1.running_var.data[first_layer_select]
        
        prev_out_channels = selected_channels.get('select_gate1', None)
        
        for layer_idx in range(1, 5):
            layer_name = f'layer{layer_idx}'
            if hasattr(model, layer_name):
                old_layer = getattr(model, layer_name)
                new_layer = getattr(new_model, layer_name)
                
                for block_idx, (old_block, new_block) in enumerate(zip(old_layer, new_layer)):
                    block_name = f'{layer_name}.{block_idx}'
                    current_in_channels = prev_out_channels
                    current_out_channels = selected_channels.get(f'{block_name}.select_gate2', None)
                    
                    if current_in_channels is not None and current_out_channels is not None:
                        new_block.conv1.weight.data = old_block.conv1.weight.data[current_out_channels][:, current_in_channels]
                        new_block.bn1.weight.data = old_block.bn1.weight.data[current_out_channels]
                        new_block.bn1.bias.data = old_block.bn1.bias.data[current_out_channels]
                        new_block.bn1.running_mean.data = old_block.bn1.running_mean.data[current_out_channels]
                        new_block.bn1.running_var.data = old_block.bn1.running_var.data[current_out_channels]
                        
                        new_block.conv2.weight.data = old_block.conv2.weight.data[current_out_channels][:, current_out_channels]
                        new_block.bn2.weight.data = old_block.bn2.weight.data[current_out_channels]
                        new_block.bn2.bias.data = old_block.bn2.bias.data[current_out_channels]
                        new_block.bn2.running_mean.data = old_block.bn2.running_mean.data[current_out_channels]
                        new_block.bn2.running_var.data = old_block.bn2.running_var.data[current_out_channels]
                        
                        if hasattr(old_block.shortcut, '0'):
                            shortcut_conv = old_block.shortcut[0]
                            new_shortcut_conv = new_block.shortcut[0]
                            new_shortcut_conv.weight.data = shortcut_conv.weight.data[current_out_channels][:, current_in_channels]
                            
                            if len(old_block.shortcut) > 1:
                                shortcut_bn = old_block.shortcut[1]
                                new_shortcut_bn = new_block.shortcut[1]
                                new_shortcut_bn.weight.data = shortcut_bn.weight.data[current_out_channels]
                                new_shortcut_bn.bias.data = shortcut_bn.bias.data[current_out_channels]
                                new_shortcut_bn.running_mean.data = shortcut_bn.running_mean.data[current_out_channels]
                                new_shortcut_bn.running_var.data = shortcut_bn.running_var.data[current_out_channels]
                    
                    prev_out_channels = current_out_channels
        
        if hasattr(model, 'fc') and prev_out_channels is not None:
            new_model.fc.weight.data = model.fc.weight.data[:, prev_out_channels]
            new_model.fc.bias.data = model.fc.bias.data.clone()
    
    for name, module in new_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            shape_comparison[name]['new'] = {
                'weight': list(module.weight.shape),
                'type': module.__class__.__name__
            }
    
    return new_model

def select_subnet(model, loader, device, target_sparsity=0.5):
  
    correlation_matrices = analyze_channel_correlation(model, loader, device)
    
    shape_comparison = record_shapes(model)
    
    selected_channels = select_channels_with_correlation(
        model, correlation_matrices, target_sparsity)
    
    new_model = create_pruned_model(model, selected_channels, shape_comparison)
    
    return new_model, selected_channels, shape_comparison

def evaluate_subnets(model, testloader, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(testloader)
        accuracy = 100. * correct / total
        return accuracy

def print_shape_comparison(shape_comparison):
    print("\nLayer Shape Comparison (Original vs Pruned):")
    print("-" * 80)
    print(f"{'Layer Name':<30} {'Original Shape':<25} {'Pruned Shape':<25}")
    print("-" * 80)

    total_params_original = 0
    total_params_pruned = 0
    
    layer_channels = {
        'conv1': None,
        'layer1': [None, None],
        'layer2': [None, None],
        'layer3': [None, None],
        'layer4': [None, None]
    }
    
    def count_conv_params(shape):
        # [out_channels, in_channels, kernel_h, kernel_w]
        return np.prod(shape)         
 
    def count_bn_params(num_features):
        # weight, bias, running_mean, running_var
        return num_features * 4
        
    def count_linear_params(shape):
        # weight + bias
        return np.prod(shape) + shape[0]
    
    def format_shape(shape):
        return f"{list(shape)}"
    
    for name, shapes in sorted(shape_comparison.items()):
        if 'original' in shapes and 'new' in shapes:
            orig_shape = shapes['original']['weight']
            new_shape = shapes['new']['weight']
            layer_type = shapes['original']['type']
            
            if layer_type == 'Conv2d':
                orig_params = count_conv_params(orig_shape)
                new_params = count_conv_params(new_shape)
            elif layer_type == 'BatchNorm2d':
                orig_params = count_bn_params(orig_shape[0])
                new_params = count_bn_params(new_shape[0])
            elif layer_type == 'Linear':
                orig_params = count_linear_params(orig_shape)
                new_params = count_linear_params(new_shape)
            

            
            total_params_original += orig_params
            total_params_pruned += new_params

            reduction = (1 - new_params/orig_params) * 100

            print(f"{name:<30} {str(orig_shape):<25} {str(new_shape):<25}")
            print(f"{'Parameters:':<30} {orig_params:,} -> {new_params:,} ({reduction:.1f}% reduction)")
            print("-" * 80)
            
            if layer_type == 'Conv2d':
                if 'conv1' in name and 'layer' not in name:
                    layer_channels['conv1'] = new_shape[0]
                elif 'layer1' in name:
                    if 'conv1' in name:
                        layer_channels['layer1'][0] = new_shape[0]
                    elif 'conv2' in name:
                        layer_channels['layer1'][1] = new_shape[0]
                elif 'layer2' in name:
                    if 'conv1' in name:
                        layer_channels['layer2'][0] = new_shape[0]
                    elif 'conv2' in name:
                        layer_channels['layer2'][1] = new_shape[0]
                elif 'layer3' in name:
                    if 'conv1' in name:
                        layer_channels['layer3'][0] = new_shape[0]
                    elif 'conv2' in name:
                        layer_channels['layer3'][1] = new_shape[0]
                elif 'layer4' in name:
                    if 'conv1' in name:
                        layer_channels['layer4'][0] = new_shape[0]
                    elif 'conv2' in name:
                        layer_channels['layer4'][1] = new_shape[0]
    
    if layer_channels['conv1'] is None:
        layer_channels['conv1'] = 64  
    
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if None in layer_channels[layer_name]:
            if layer_channels[layer_name][0] is not None:
                layer_channels[layer_name][1] = layer_channels[layer_name][0]
            elif layer_channels[layer_name][1] is not None:
                layer_channels[layer_name][0] = layer_channels[layer_name][1]
            else:
                if layer_name == 'layer1':
                    layer_channels[layer_name] = [layer_channels['conv1']] * 2
                else:
                    prev_layer = f'layer{int(layer_name[-1])-1}'
                    layer_channels[layer_name] = [layer_channels[prev_layer][1]] * 2
    


    print("\nPruned Channel Configuration:")
    print("layer_channels = {")
    print(f"    'conv1': {layer_channels['conv1']},")
    print(f"    'layer1': {layer_channels['layer1']},")
    print(f"    'layer2': {layer_channels['layer2']},")
    print(f"    'layer3': {layer_channels['layer3']},")
    print(f"    'layer4': {layer_channels['layer4']}")
    print("}")
    
    def calculate_resnet18_params(channels):
        total = 0
        
        # First conv layer
        total += 3 * channels['conv1'] * 3 * 3  # conv1
        total += channels['conv1'] * 4  # bn1
        
        # Layer1 (2 blocks)
        total += channels['conv1'] * channels['layer1'][0] * 3 * 3  # block1.conv1
        total += channels['layer1'][0] * 4  # block1.bn1
        total += channels['layer1'][0] * channels['layer1'][1] * 3 * 3  # block1.conv2
        total += channels['layer1'][1] * 4  # block1.bn2
        
        total += channels['layer1'][1] * channels['layer1'][0] * 3 * 3  # block2.conv1
        total += channels['layer1'][0] * 4  # block2.bn1
        total += channels['layer1'][0] * channels['layer1'][1] * 3 * 3  # block2.conv2
        total += channels['layer1'][1] * 4  # block2.bn2
        
        # Layer2 (2 blocks)
        total += channels['layer1'][1] * channels['layer2'][0] * 3 * 3  # block1.conv1
        total += channels['layer2'][0] * 4  # block1.bn1
        total += channels['layer2'][0] * channels['layer2'][1] * 3 * 3  # block1.conv2
        total += channels['layer2'][1] * 4  # block1.bn2
        total += channels['layer1'][1] * channels['layer2'][1] * 1 * 1  # shortcut
        total += channels['layer2'][1] * 4  # shortcut bn
        
        total += channels['layer2'][1] * channels['layer2'][0] * 3 * 3  # block2.conv1
        total += channels['layer2'][0] * 4  # block2.bn1
        total += channels['layer2'][0] * channels['layer2'][1] * 3 * 3  # block2.conv2
        total += channels['layer2'][1] * 4  # block2.bn2
        
        # Layer3 (2 blocks)
        total += channels['layer2'][1] * channels['layer3'][0] * 3 * 3  # block1.conv1
        total += channels['layer3'][0] * 4  # block1.bn1
        total += channels['layer3'][0] * channels['layer3'][1] * 3 * 3  # block1.conv2
        total += channels['layer3'][1] * 4  # block1.bn2
        total += channels['layer2'][1] * channels['layer3'][1] * 1 * 1  # shortcut
        total += channels['layer3'][1] * 4  # shortcut bn
        
        total += channels['layer3'][1] * channels['layer3'][0] * 3 * 3  # block2.conv1
        total += channels['layer3'][0] * 4  # block2.bn1
        total += channels['layer3'][0] * channels['layer3'][1] * 3 * 3  # block2.conv2
        total += channels['layer3'][1] * 4  # block2.bn2
        
        # Layer4 (2 blocks)
        total += channels['layer3'][1] * channels['layer4'][0] * 3 * 3  # block1.conv1
        total += channels['layer4'][0] * 4  # block1.bn1
        total += channels['layer4'][0] * channels['layer4'][1] * 3 * 3  # block1.conv2
        total += channels['layer4'][1] * 4  # block1.bn2
        total += channels['layer3'][1] * channels['layer4'][1] * 1 * 1  # shortcut
        total += channels['layer4'][1] * 4  # shortcut bn
        
        total += channels['layer4'][1] * channels['layer4'][0] * 3 * 3  # block2.conv1
        total += channels['layer4'][0] * 4  # block2.bn1
        total += channels['layer4'][0] * channels['layer4'][1] * 3 * 3  # block2.conv2
        total += channels['layer4'][1] * 4  # block2.bn2
        
        # FC layer
        total += channels['layer4'][1] * 10 + 10  # weight + bias
        
        return total
    
    calculated_params = calculate_resnet18_params(layer_channels)
    print(f"\nCalculated total parameters: {calculated_params:,}")
    print(f"\nOrign total parameters: {total_params_pruned:,}")
    reduction = (1 - calculated_params/total_params_pruned) * 100
    print(f"Reduction       : {reduction:.1f}%")

    
    return layer_channels

def create_split_datasets(stage1_classes, stage2_classes, allow_overlap=True):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    stage1_label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(set(stage1_classes)))}
    stage2_label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(set(stage2_classes)))}

    class LabelRemappedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, selected_classes, label_map):
            self.dataset = dataset
            self.indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
            self.label_map = label_map

        def __getitem__(self, idx):
            image, label = self.dataset[self.indices[idx]]
            return image, self.label_map[label]

        def __len__(self):
            return len(self.indices)

    stage1_trainset = LabelRemappedDataset(trainset, stage1_classes, stage1_label_map)
    stage1_testset = LabelRemappedDataset(testset, stage1_classes, stage1_label_map)
    stage2_trainset = LabelRemappedDataset(trainset, stage2_classes, stage2_label_map)
    stage2_testset = LabelRemappedDataset(testset, stage2_classes, stage2_label_map)

    stage1_trainloader = DataLoader(
        stage1_trainset, batch_size=128, shuffle=True, num_workers=2)
    stage1_testloader = DataLoader(
        stage1_testset, batch_size=100, shuffle=False, num_workers=2)
    
    stage2_trainloader = DataLoader(
        stage2_trainset, batch_size=128, shuffle=True, num_workers=2)
    stage2_testloader = DataLoader(
        stage2_testset, batch_size=100, shuffle=False, num_workers=2)

    return (stage1_trainloader, stage1_testloader), (stage2_trainloader, stage2_testloader)
    
def main_pruning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

       
    model = ResNet18(num_classes=10, sparsity_target=0.5)
    checkpoint = torch.load('Resnet18_Cifar10_stage2.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

  
    model = model.to(device)
    model.eval()

    model.set_training_phase('gates')

    original_acc = evaluate_subnets(model, testloader, device)  
    print(f"Original Model Accuracy: {original_acc:.2f}%")

    target_sparsity = 0.1
    print(f"\nSelecting subnet with target sparsity {target_sparsity}")
    subnet, selected_channels, shape_comparison = select_subnet(
        model, testloader, device, target_sparsity)
    
    subnet = subnet.to(device)
    subnet_acc = evaluate_subnets(subnet, testloader, device)
    print(f"Subnet Accuracy: {subnet_acc:.2f}%")

        
    layer_channels = print_shape_comparison(shape_comparison)
    
    torch.save({
        'model_state_dict': subnet.state_dict(),
        'selected_channels': selected_channels,
        'shape_comparison': shape_comparison,
        'layer_channels': layer_channels,  
        'accuracy': subnet_acc
    }, 'Resnet18_Cifar10_subnet.pth')
    



def main_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    model = ResNet18(num_classes=10, sparsity_target=0.5)
    
    trainer = TwoStageTrainer(
        model=model,
        train_loader=trainloader,
        test_loader=testloader,
        device=device,
        learning_rate=0.001,
        weight_decay=5e-4
    )

    try:
        print("\nStarting Stage 1 - Training ResNet backbone")
        stage1_metrics = trainer.train_stage1(epochs=100)
        print(f"\nStage 1 completed! Best accuracy: {trainer.best_acc['stage1']:.2f}%")

        print("\nStarting Stage 2 - Joint training")
        stage2_metrics = trainer.train_stage2(epochs=100)
        print(f"\nStage 2 completed! Best accuracy: {trainer.best_acc['stage2']:.2f}%")

        print("\nTraining completed successfully!")
        print(f"Stage 1 best accuracy: {trainer.best_acc['stage1']:.2f}%")
        print(f"Stage 2 best accuracy: {trainer.best_acc['stage2']:.2f}%")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    main_train()
    main_pruning()