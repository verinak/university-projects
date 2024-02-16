# Operating Systems

## Assignments

### Least Recently Used (LRU) Algorithm Implementation
Implementation of LRU algorithm using Java to simulate Page Replacement in Android.

### Multilevel Queue Implementation
Implementation of Multilevel Queue (MLQ) CPU Scheduling using Python<br/>
The MLQ has three queues:
- First the high queue is represented by 0 in the priority column, and the processes are scheduled by Round Robin scheduling algorithms with q = 3 in this queue.
- Second the medium queue is represented by 1 in the priority column, and the processes are scheduled by Shortest Remaining First scheduling algorithms with q = 2 in this queue.
- Third the low queue is represented by 2 in the priority column, and the processes are scheduled by Shortest Job Next scheduling algorithms in this queue.

Bonus if you do GUI that print Kernel Diagram and Ready Queue in Table

### Clock Algorithm Implementation
Implementation of Second Chance/Clock Page Replacement Policy using Python

## Final Project
File System implementation in Java

Types of File Systems:
- Single hierarchy with one top-level root directory (e.g., Linux).
- Multiple distinct file hierarchies, each with its own top-level root directory (e.g., Windows C: or D:).

Types of Files:
- File:
    - Attributes: Name, Identifier, Location, Type, Size, Blocks, Protection, Creation Time, Modification Time, Access Time, Content.
    - Actions and Functions: Create, Read, Write, Copy, Move/Rename, Delete, Get Information, Change Permissions.
- Directory:
    - Attributes: Type, Children (files and subdirectories).
    - Actions and Functions: Create, Delete, List Contents, Get Information, Change Permissions, Search.
- Root Node (Partition):
    - Attributes: Partition Label/Name, UUID, Size, Used Space, Free Space, Block Size.