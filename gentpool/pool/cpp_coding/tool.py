### Define your custom tool here. Check prebuilts in gentopia.tool (:###
from gentopia.tools import *

import subprocess
import tempfile
import os
import shutil

class CppCodeInterpreter:
    def __init__(self, timeout=300):
        self.timeout = timeout

    def execute_code(self, code):
        # Remove leading and trailing whitespace (including newlines)
        code = code.strip()

        # Check if the code starts with '```cpp' and ends with '```', if so, remove them
        if code.startswith('```') and code.endswith('```'):
            code_lines = code.split('\n')
            code = '\n'.join(code_lines[1:-1])

        # Create a temporary directory to hold the .cpp file and output
        temp_dir = tempfile.mkdtemp()
        try:
            # Write the C++ code to a .cpp file in the temp directory
            cpp_file_path = os.path.join(temp_dir, 'temp.cpp')
            with open(cpp_file_path, 'w') as f:
                f.write(code)
            
            # Compile the C++ code into an executable
            compile_cmd = f"g++ {cpp_file_path} -o {temp_dir}/temp.out"
            compile_process = subprocess.run(compile_cmd, shell=True, timeout=self.timeout,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # If the compilation process returned a non-zero exit code, there was a compile error
            if compile_process.returncode != 0:
                error_message = compile_process.stderr.decode('utf-8')
                # Only return the first line of the error
                return f"Compile error: {error_message.splitlines()[0]}"
            
            # Execute the compiled code
            execute_cmd = f"{temp_dir}/temp.out"
            execute_process = subprocess.run(execute_cmd, shell=True, timeout=self.timeout,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # If the execution process returned a non-zero exit code, there was a runtime error
            if execute_process.returncode != 0:
                error_message = execute_process.stderr.decode('utf-8')
                # Only return the first line of the error
                return f"Runtime error: {error_message.splitlines()[0]}"

            output_message = execute_process.stdout.decode('utf-8')
            # Only return the first 1024 characters of the output
            return output_message[:512]

        except Exception as e:
            # Only return the type of the exception, not the entire error message
            return f"System error: {type(e).__name__}"
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)


class CppCodeInterpreterTool(BaseTool):
    """C++ Code Interpreter Tool"""
    name = "cpp_code_interpreter"
    description = "A tool to execute C++ code and retrieve the command line output. Input should be executable C++ code."
    args_schema: Optional[Type[BaseModel]] = create_model("CodeInterpreter", code=(str, ...))
    interpreter = CppCodeInterpreter()

    def _run(self, code: AnyStr) -> Any:
        return self.interpreter.execute_code(code)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
if __name__ == "__main__":
    tool = CppCodeInterpreterTool()
    ans = tool._run("""
    #include<iostream>
    int main() {
        int n = 10, t1 = 0, t2 = 1, nextTerm = 0;
        for (int i = 1; i <= n; ++i) {
            if(i == 1) {
                std::cout << t1 << ", ";
                continue;
            }
            if(i == 2) {
                std::cout << t2 << ", ";
                continue;
            }
            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;
            std::cout << nextTerm << ", ";
        }
        return 0;
    }
    """)
    print(ans)
