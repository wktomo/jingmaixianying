
import sys
import re
from pathlib import Path

def find_vessel_controlnet():
    """查找 vessel_controlnet.py 文件"""
    # 常见路径
    possible_paths = [
        Path("models/enhancement/vessel_controlnet.py"),
        Path("vessel_controlnet.py"),
        Path("../models/enhancement/vessel_controlnet.py"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # 如果没找到，搜索当前目录
    for path in Path(".").rglob("vessel_controlnet.py"):
        return path
    
    return None

def analyze_forward_method(file_path):
    """分析 forward 方法"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("="*60)
    print("分析 VesselDiffusionEnhancer 类")
    print("="*60)
    
    # 查找 forward 方法
    forward_pattern = r'def forward\(self[^)]*\):'
    matches = list(re.finditer(forward_pattern, content))
    
    if matches:
        print(f"\n✓ 找到 {len(matches)} 个 forward 方法\n")
        
        for i, match in enumerate(matches, 1):
            # 获取方法签名
            start = match.start()
            # 找到方法定义前的类名
            before = content[:start].split('\n')[-20:]
            class_name = "Unknown"
            for line in reversed(before):
                if 'class ' in line:
                    class_name = line.strip()
                    break
            
            # 获取完整的方法签名
            sig_start = start
            sig_end = content.find(':', start) + 1
            signature = content[sig_start:sig_end]
            
            print(f"方法 {i}:")
            print(f"  所属类: {class_name}")
            print(f"  签名: {signature}")
            
            # 检查是否有 encoder_hidden_states 参数
            if 'encoder_hidden_states' in signature:
                print("  ✓ 已包含 encoder_hidden_states 参数")
            else:
                print("  ✗ 缺少 encoder_hidden_states 参数 - 这可能是问题所在")
            print()
    
    # 查找 UNet 调用
    print("="*60)
    print("分析 UNet 调用")
    print("="*60)
    
    unet_pattern = r'self\.unet\('
    unet_matches = list(re.finditer(unet_pattern, content))
    
    if unet_matches:
        print(f"\n✓ 找到 {len(unet_matches)} 处 UNet 调用\n")
        
        for i, match in enumerate(unet_matches, 1):
            # 获取调用上下文（前后5行）
            pos = match.start()
            lines_before = content[:pos].split('\n')
            line_num = len(lines_before)
            
            # 获取完整的函数调用（找到匹配的括号）
            call_start = pos
            call_end = pos
            paren_count = 0
            in_call = False
            
            for j, char in enumerate(content[pos:pos+1000]):
                if char == '(':
                    paren_count += 1
                    in_call = True
                elif char == ')':
                    paren_count -= 1
                    if in_call and paren_count == 0:
                        call_end = pos + j + 1
                        break
            
            call_text = content[call_start:call_end]
            
            print(f"调用 {i} (行号约 {line_num}):")
            print(f"  代码:")
            for line in call_text.split('\n'):
                print(f"    {line}")
            
            # 检查是否有 encoder_hidden_states
            if 'encoder_hidden_states' in call_text:
                print("  ✓ 已传入 encoder_hidden_states")
            else:
                print("  ✗ 未传入 encoder_hidden_states - 需要添加！")
            print()
    
    # 查找 null_text_embeds 或类似的初始化
    print("="*60)
    print("检查是否已有空文本嵌入")
    print("="*60)
    
    if 'null_text' in content.lower() or 'empty_text' in content.lower():
        print("\n✓ 代码中已经有相关逻辑")
    else:
        print("\n✗ 未找到空文本嵌入的初始化")
        print("  建议: 需要添加 _init_null_text_embeddings() 方法")
    
    return True

def main():
    print("\n" + "🔍 vessel_controlnet.py 诊断工具")
    print("="*60)
    
    # 查找文件
    print("\n正在查找 vessel_controlnet.py...")
    file_path = find_vessel_controlnet()
    
    if not file_path:
        print("❌ 未找到 vessel_controlnet.py 文件")
        print("\n请确保在项目根目录运行此脚本，或手动指定文件路径：")
        print("  python diagnose.py /path/to/vessel_controlnet.py")
        
        if len(sys.argv) > 1:
            file_path = Path(sys.argv[1])
            if not file_path.exists():
                print(f"❌ 指定的文件不存在: {file_path}")
                return
        else:
            return
    
    print(f"✓ 找到文件: {file_path}")
    
    # 分析文件
    try:
        analyze_forward_method(file_path)
    except Exception as e:
        print(f"\n❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 提供建议
    print("\n" + "="*60)
    print("📋 修复建议")
    print("="*60)
    print("""
如果诊断显示缺少 encoder_hidden_states:

1. 在 __init__ 方法中添加:
   self.null_text_embeds = self._init_null_text_embeddings()

2. 添加新方法:
   def _init_null_text_embeddings(self):
       # 见 FIX_MATRIX_ERROR.md 中的完整代码
       ...

3. 在 self.unet(...) 调用中添加:
   encoder_hidden_states=self.null_text_embeds.repeat(batch_size, 1, 1)

详细步骤请查看: FIX_MATRIX_ERROR.md
    """)
    
    print("\n如果需要帮助，请将 vessel_controlnet.py 文件提供给我，")
    print("我可以帮你精确定位并修改。")

if __name__ == "__main__":
    main()