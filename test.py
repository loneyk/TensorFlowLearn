# def BinarySearch(lista, key):
#     min = 0
#     max = len(lista) - 1
#
#     if key in lista:
#         while True:
#             mid = int((min + max) / 2)
#             if lista[mid] > key:
#                 max = mid - 1
#             elif lista[mid] < key:
#                 min = mid + 1
#             elif lista[mid] == key:
#                 print(str(mid))
#                 return lista[mid]
#     else:
#         print("None")
#
#
# if __name__ == "__main__":
#     arr = [2, 5, 8, 13, 49, 80]
#     while True:
#         key = input("请输入你要查找的数字：")
#         if key == " ":
#             print("谢谢使用！")
#             break
#         else:
#             BinarySearch(arr, int(key))

def fib(n):
    # if n==1 or n==2:
    #     return 1
    # return fib(n-1) + fib(n-2)

    # a, b = 1, 1
    # for i in range(n-1):
    #     a, b = b, a+b
    # return a
    if n == 1:
        return 1
    if n == 2:
        return [1, 1]
    fibs = [1, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs




print(fib(10)[9])











