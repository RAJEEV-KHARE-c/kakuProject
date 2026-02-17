#!/usr/bin/env python3
"""
Fix syntax error in app.py - remove duplicate finally blocks
"""

def fix_syntax_error():
    print("🔧 Fixing syntax error in app.py...")
    
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix the duplicate finally blocks around line 647-651
    fixed_lines = []
    i = 0
    while i < len(lines):
        if (i < len(lines) - 5 and 
            'finally:' in lines[i] and 
            'conn.close()' in lines[i+1] and 
            'return False' in lines[i+2] and
            'finally:' in lines[i+3] and
            'conn.close()' in lines[i+4]):
            
            print(f"📍 Found duplicate finally blocks at line {i+1}")
            # Replace with single finally block
            fixed_lines.append('        finally:\n')
            fixed_lines.append('            conn.close()\n')
            i += 5  # Skip the duplicate lines
        else:
            fixed_lines.append(lines[i])
            i += 1
    
    # Write fixed content back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Syntax error fixed!")

if __name__ == "__main__":
    fix_syntax_error()