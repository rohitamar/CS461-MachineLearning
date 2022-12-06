#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// Struct representing individual blocks of memory, Header | Payload | Footer
typedef struct Block {
    size_t blockSize;
    size_t memSize;
    struct Block *next;
    struct Block *prev;
    unsigned char *buf;
} Block;

// Library Variables
int allocAlgo = 0;
Block* freeList;
Block* allocList;
Block* heap;
Block* nextPtr;
int heapSize = 1024*1024;
int memUsed = 0;
int spaceUsed = 0;

// Library Functions

// Initializes 1MB of Heap
void myinit(int allocAlg) {
    heap = NULL;
    nextPtr = NULL;
    heap = malloc(heapSize);
    freeList = heap;
    freeList->blockSize = (heapSize) - sizeof(Block);
    freeList->memSize = 0;
    freeList->next = NULL;
    freeList->prev = NULL;
    freeList->buf = (unsigned char *) freeList + sizeof(Block);
    allocList = NULL;
    nextPtr = freeList;
    allocAlgo = allocAlg;
}


void* mymalloc(size_t size) {
    if(size == 0) return NULL;
    int memUse = size;
    if(size % 8 != 0) {
        size += 8 - size % 8;
    }
    Block* curr = freeList;
    // AllocAlgo 0 = First Fit
    if(allocAlgo == 0) {
        while(curr != NULL && freeList != NULL) {

            if(size < curr->blockSize && curr->blockSize - size > sizeof(Block)) { // If block is larger than size & enough room for header
                // Initialize new free block
                Block* new = (Block*) (curr->buf + size);
                new->blockSize = curr->blockSize - sizeof(Block) - size;
                new->prev = curr->prev;
                new->next = curr->next;
                
                new->buf = (unsigned char *) new + sizeof(Block);
                // Adjust curr size
                curr->blockSize = size;

                // Handle freeList
                if(curr->prev == NULL) {
                    freeList = new;
                }

                // Remove from free list
                if(curr->prev != NULL) {
                    curr->prev->next = curr->next;
                }
                if(curr->next != NULL) {
                    curr->next->prev = curr->prev;
                }

                // Add to allocList
                if(allocList == NULL) {
                    allocList = curr;
                } else {
                    Block* ptr = allocList;
                    while(ptr->next != NULL && curr > ptr) {
                        ptr = ptr->next;
                    }
                    if(curr > ptr) {
                        curr->prev = ptr;
                        curr->next = ptr->next;
                        ptr->next = curr;
                        if(curr->next != NULL) {
                            curr->next->prev = curr;
                        }
                    } else {
                        curr->next = ptr;
                        if(ptr->prev != NULL) {
                            ptr->prev->next = curr;
                        } else {
                            allocList = curr;
                        }
                        ptr->prev = curr;

                    }
                    // printf("ptr:%p", ptr);
                    // printf("curr:%p", curr);
                    
                }
                memUsed += memUse;
                spaceUsed += size + sizeof(Block);
                curr->memSize = memUse;
                // Return ptr
                return curr->buf;
            } else if(size <= curr->blockSize) { // If block is exact size needed or reamining block has no space for header
                // Adjust first node
                if(curr->prev == NULL) {
                    freeList = curr->next;
                }

                // Remove from free list
                if(curr->prev != NULL) {
                    curr->prev->next = curr->next;
                }
                if(curr->next != NULL) {
                    curr->next->prev = curr->prev;
                }

                // Add to allocList
                if(allocList == NULL) {
                    allocList = curr;
                } else {
                    Block* ptr = allocList;
                    while(ptr->next != NULL && curr > ptr) {
                        ptr = ptr->next;
                    }
                    if(curr > ptr) {
                        curr->prev = ptr;
                        curr->next = ptr->next;
                        ptr->next = curr;
                        if(curr->next != NULL) {
                            curr->next->prev = curr;
                        }
                    } else {
                        curr->next = ptr;
                        if(ptr->prev != NULL) {
                            ptr->prev->next = curr;
                        } else {
                            allocList = curr;
                        }
                        ptr->prev = curr;

                    }
                }
                memUsed += memUse;
                spaceUsed += size + sizeof(Block);
                curr->memSize = memUse;
                // Return ptr
                return curr->buf;
            } 
            curr = curr->next;
        }

    } else if(allocAlgo == 1) { // Next Fit
        if(freeList == NULL) return NULL;
        Block* initPtr = nextPtr;
        int loop = 0;
        while(nextPtr != NULL && freeList != NULL) {
            // check for loop
            if(nextPtr == initPtr) loop++;
            if(loop == 2) break;

            if(size < nextPtr->blockSize && nextPtr->blockSize - size > sizeof(Block)) { // If block is larger than size & has room for header
                // Initialize new free block
                Block* new = (Block*) (nextPtr->buf + size);
                new->blockSize = nextPtr->blockSize - sizeof(Block) - size;
                new->prev = nextPtr->prev;
                new->next = nextPtr->next;
                
                new->buf = (unsigned char *) new + 40;
                // Adjust curr size
                nextPtr->blockSize = size;

                // Handle freeList
                if(nextPtr->prev == NULL) {
                    freeList = new;
                }

                // Remove from free list
                if(nextPtr->prev != NULL) {
                    nextPtr->prev->next = nextPtr->next;
                }
                if(nextPtr->next != NULL) {
                    nextPtr->next->prev = nextPtr->prev;
                }

                // Add to allocList
                Block* newBlock = nextPtr;
                if(allocList == NULL) {
                    allocList = newBlock;
                } else {
                    Block* ptr = allocList;
                    while(ptr->next != NULL && newBlock > ptr) {
                        ptr = ptr->next;
                    }
                    if(newBlock > ptr) {
                        newBlock->prev = ptr;
                        newBlock->next = ptr->next;
                        ptr->next = newBlock;
                        if(newBlock->next != NULL) {
                            newBlock->next->prev = newBlock;
                        }
                    } else {
                        newBlock->next = ptr;
                        if(ptr->prev != NULL) {
                            ptr->prev->next = newBlock;
                        } else {
                            allocList = newBlock;
                        }
                        ptr->prev = newBlock;

                    }
                }

                // move nextPtr accordingly
                if(nextPtr->next == NULL) {
                    nextPtr = freeList;
                } else {
                    nextPtr = nextPtr->next;
                }

                memUsed += memUse;
                spaceUsed += size + sizeof(Block);
                newBlock->memSize = memUse;
                // Return ptr
                return newBlock->buf;
            } else if(size <= nextPtr->blockSize) { // If block is exact size needed or reamining memory is not big enough for header
                // Adjust first node

                if(nextPtr->prev == NULL) {
                    freeList = nextPtr->next;
                }

                // Remove from free list
                if(nextPtr->prev != NULL) {
                    nextPtr->prev->next = nextPtr->next;
                }
                if(nextPtr->next != NULL) {
                    nextPtr->next->prev = nextPtr->prev;
                }

                // Add to allocList
                Block* newBlock = nextPtr;
                if(allocList == NULL) {
                    allocList = newBlock;
                } else {
                    Block* ptr = allocList;
                    while(ptr->next != NULL && newBlock > ptr) {
                        ptr = ptr->next;
                    }
                    if(newBlock > ptr) {
                        newBlock->prev = ptr;
                        newBlock->next = ptr->next;
                        ptr->next = newBlock;
                        if(newBlock->next != NULL) {
                            newBlock->next->prev = newBlock;
                        }
                    } else {
                        newBlock->next = ptr;
                        if(ptr->prev != NULL) {
                            ptr->prev->next = newBlock;
                        } else {
                            allocList = newBlock;
                        }
                        ptr->prev = newBlock;

                    }
                }

                // move nextPtr accordingly
                if(nextPtr->next == NULL) {
                    nextPtr = freeList;
                } else {
                    nextPtr = nextPtr->next;
                }

                memUsed += memUse;
                spaceUsed += size + sizeof(Block);
                newBlock->memSize = memUse;
                // Return ptr
                return newBlock->buf;
            }
            if(nextPtr->next == NULL) {
                nextPtr = freeList;
            } else {
                nextPtr = nextPtr->next;
            }
        }
    } else if(allocAlgo == 2) { // Best Fit
        if(freeList == NULL) return NULL;
        Block* minBlock = freeList;
        while(curr != NULL && freeList != NULL) {
            if(curr->blockSize < minBlock->blockSize && curr->blockSize >= size) {
                minBlock = curr;
            }
            curr = curr->next;
        }


        if(size < minBlock->blockSize && minBlock->blockSize - size > sizeof(Block)) { // If block is larger than size & enough room for header
            // Initialize new free block
            Block* new = (Block*) (minBlock->buf + size);
            new->blockSize = minBlock->blockSize - sizeof(Block) - size;
            new->prev = minBlock->prev;
            new->next = minBlock->next;
            
            new->buf = (unsigned char *) new + sizeof(Block);
            // Adjust curr size
            minBlock->blockSize = size;

            // Handle freeList
            if(minBlock->prev == NULL) {
                freeList = new;
            }

            // Remove from free list
            if(minBlock->prev != NULL) {
                minBlock->prev->next = minBlock->next;
            }
            if(minBlock->next != NULL) {
                minBlock->next->prev = minBlock->prev;
            }

            // Add to allocList
            if(allocList == NULL) {
                allocList = minBlock;
            } else {
                Block* ptr = allocList;
                while(ptr->next != NULL && minBlock > ptr) {
                    ptr = ptr->next;
                }
                if(minBlock > ptr) {
                    minBlock->prev = ptr;
                    minBlock->next = ptr->next;
                    ptr->next = minBlock;
                    if(minBlock->next != NULL) {
                        minBlock->next->prev = minBlock;
                    }
                } else {
                    minBlock->next = ptr;
                    if(ptr->prev != NULL) {
                        ptr->prev->next = minBlock;
                    } else {
                        allocList = minBlock;
                    }
                    ptr->prev = minBlock;
                }
            }
            memUsed += memUse;
            spaceUsed += size + sizeof(Block);
            minBlock->memSize = memUse;
            // Return ptr
            return minBlock->buf;
        } else if(size <= minBlock->blockSize) { // If block is exact size needed or reamining memory is not big enough for header
            // Adjust first node
            if(minBlock->prev == NULL) {
                freeList = minBlock->next;
            }

            // Remove from free list
            if(minBlock->prev != NULL) {
                minBlock->prev->next = minBlock->next;
            }
            if(minBlock->next != NULL) {
                minBlock->next->prev = minBlock->prev;
            }

            // Add to allocList
            if(allocList == NULL) {
                allocList = minBlock;
            } else {
                Block* ptr = allocList;
                while(ptr->next != NULL && minBlock > ptr) {
                    ptr = ptr->next;
                }
                if(minBlock > ptr) {
                    minBlock->prev = ptr;
                    minBlock->next = ptr->next;
                    ptr->next = minBlock;
                    if(minBlock->next != NULL) {
                        minBlock->next->prev = minBlock;
                    }
                } else {
                    minBlock->next = ptr;
                    if(ptr->prev != NULL) {
                        ptr->prev->next = minBlock;
                    } else {
                        allocList = minBlock;
                    }
                    ptr->prev = minBlock;
                }
            }
            memUsed += memUse;
            spaceUsed += size + sizeof(Block);
            minBlock->memSize = memUse;
            // Return ptr
            return minBlock->buf;
        }

    }

    return NULL;
}

// define coalescing function
static void coalesce(Block* curr) {
    curr->memSize = 0;
    // coalesce with next block
    if(curr->next != NULL && curr->buf + curr->blockSize == (void*)curr->next) {
        curr->blockSize = curr->blockSize + curr->next->blockSize + sizeof(Block);
        curr->next = curr->next->next;
        if(curr->next != NULL) {
            curr->next->prev = curr;
        }
    }

    // coalesce with prev block
    if(curr->prev != NULL && curr->prev->buf + curr->prev->blockSize == (void*)curr) {
        curr->prev->blockSize = curr->prev->blockSize + curr->blockSize + sizeof(Block);
        curr->prev->next = curr->next;
        if(curr->prev->next != NULL) {
            curr->prev->next->prev = curr->prev;
        }
    }
}

void myfree(void* ptr) {
    if(ptr != NULL) {
        if(ptr < (void *)heap || ptr > (void *)heap + heapSize) printf("error: not a heap pointer\n");
        else {
            // Check for double free
            Block* tmp = freeList;
            while(tmp != NULL) {
                if(tmp->buf == ptr) {
                    printf("error: double free\n");
                    return;
                }
                tmp = tmp->next;
            }

            // Check if ptr was returned by mymalloc call
            tmp = allocList;
            bool found = false;

            while(tmp != NULL) {
                if(tmp->buf == ptr) {
                    found = true;

                    if(freeList == NULL) {
                        // remove from allocList
                        memUsed -= tmp->memSize;
                        spaceUsed -= tmp->blockSize + sizeof(Block);
                        if(tmp->prev != NULL) {
                            tmp->prev->next = tmp->next;
                        } else {
                            allocList = tmp->next;
                        }
                        if(tmp->next != NULL) {
                            tmp->next->prev = tmp->prev;
                        }
                        // add to freelist
                        tmp->next = NULL;
                        tmp->prev = NULL;
                        freeList = tmp;
                        break;
                    }
                    // search for appropriate spot in free list
                    Block* tmp2 = freeList;
                    while(tmp2->next != NULL && tmp->buf > tmp2->buf) {
                        tmp2 = tmp2->next;
                    }
                    
                    // insert block
                    // check if at start of list
                    if(tmp2->buf > tmp->buf){ // insert before
                        // remove from allocList
                        memUsed -= tmp->memSize;
                        spaceUsed -= tmp->blockSize + sizeof(Block);
                        if(tmp->prev != NULL) {
                            tmp->prev->next = tmp->next;
                        } else {
                            allocList = tmp->next;
                        }
                        if(tmp->next != NULL) {
                            tmp->next->prev = tmp->prev;
                        }
                        // add to freelist
                        if(tmp2->prev == NULL) {
                            freeList = tmp;
                            tmp->next = tmp2;
                            tmp->prev = NULL;
                            tmp2->prev = tmp;
                        } else {    
                            tmp2->prev->next = tmp;
                            tmp->prev = tmp2->prev;
                            tmp->next = tmp2;
                            tmp2->prev = tmp;
                        }

                        
                        break;
                    } else if(tmp2 == freeList && tmp->buf > tmp2->buf) { // insert after
                        // remove from allocList
                        memUsed -= tmp->memSize;
                        spaceUsed -= tmp->blockSize + sizeof(Block);
                        if(tmp->prev != NULL) {
                            tmp->prev->next = tmp->next;
                        } else {
                            allocList = tmp->next;
                        }
                        if(tmp->next != NULL) {
                            tmp->next->prev = tmp->prev;
                        }
                        // add to freelist
                        tmp->next = tmp2->next;
                        if(tmp2->next != NULL) {
                            tmp2->next->prev = tmp;
                        }
                        tmp2->next = tmp;
                        tmp->prev = tmp2;
                        //freeList = tmp;
                        break;
                    }
                }
                tmp = tmp->next;
            }
            if(found) {
                coalesce(tmp);
                return;
            } else {
                printf("error: not a malloced address\n");
                return;
            }
        }
    }
    return;
}



void* myrealloc(void* ptr, size_t size) {
    int memUse = size;
    if(size % 8 != 0) {
        size += 8 - size % 8;
    }
    if(ptr == NULL && size == 0) return NULL;
    else if(ptr == NULL) return mymalloc(size);
    else if(size == 0) {
        myfree(ptr);
        return NULL;
    }
    // get block of ptr
    // printf("ptr: %p\n", ptr);
    Block* blk = (Block*) (ptr - sizeof(Block));
    // printf("%p\n", &blk);
    // printf("%p\n", blk);

    // Realloc in current space if possible
    Block* curr = freeList;
    while(curr != NULL) {
        // space to the right of allocated blk is free
        if(blk->buf + blk->blockSize == (void*)curr && blk->blockSize + curr->blockSize + sizeof(Block) == size) {

            // Adjust first node
            if(curr->prev == NULL) {
                freeList = curr->next;
            }

            // Remove from free list
            if(curr->prev != NULL) {
                curr->prev->next = curr->next;
            }
            if(curr->next != NULL) {
                curr->next->prev = curr->prev;
            }

            memUsed += memUse - blk->memSize;
            spaceUsed += size - blk->blockSize;
            blk->memSize = memUse;
            blk->blockSize = size;

            return blk->buf;
        } else if(blk->buf + blk->blockSize == (void*)curr && blk->blockSize + curr->blockSize + sizeof(Block) > size) {
            // Initialize new free block
            Block* prev = curr->prev;
            Block* next = curr->next;
            Block* new = (Block*) (blk->buf + size);
            new->blockSize = blk->blockSize + curr->blockSize + sizeof(Block) - size;
            new->prev = prev;
            new->next = next;
            new->buf = (unsigned char *) new + sizeof(Block);

            // Handle freeList
            if(prev == NULL) {
                freeList = new;
            }

            // Remove from free list
            if(prev != NULL) {
                prev->next = new;
            }
            if(next != NULL) {
                next->prev = new;
            }

            memUsed += memUse - blk->memSize;
            spaceUsed += size - blk->blockSize;
            blk->memSize = memUse;
            blk->blockSize = size;
            return blk->buf;
        }
        curr = curr->next;
    }

    void* newPtr = mymalloc(size);
    if(newPtr != NULL) {
        memcpy(newPtr, ptr, blk->memSize);
        myfree(ptr);
        return newPtr;
    } 

    return NULL;
}

void mycleanup() {
    free(heap);
}

double utilization() {
    return (double) memUsed / spaceUsed;
}