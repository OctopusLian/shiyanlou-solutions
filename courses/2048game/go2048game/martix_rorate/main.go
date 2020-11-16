//练习矩阵的翻转操作

package main

import "fmt"

type g2048 [4][4]int

func (t *g2048) MirrorV() {
	tn := new(g2048)
	for i, line := range t {
		for j, num := range line {
			tn[len(t)-i-1][j] = num
		}
	}
	*t = *tn
}

func (t *g2048) Right90() {
	tn := new(g2048)
	for i, line := range t {
		for j, num := range line {
			tn[j][len(t)-i-1] = num
		}
	}
	*t = *tn
}

func (t *g2048) Left90() {
	tn := new(g2048)
	for i, line := range t {
		for j, num := range line {
			tn[len(line)-j-1][i] = num
		}
	}
	*t = *tn
}

func (g *g2048) R90() {
	tn := new(g2048)
	for x, line := range g {
		for y, _ := range line {
			tn[x][y] = g[len(line)-1-y][x]
		}
	}
	*g = *tn

}

func (t *g2048) Right180() {
	tn := new(g2048)
	for i, line := range t {
		for j, num := range line {
			tn[len(line)-i-1][len(line)-j-1] = num
		}
	}
	*t = *tn
}

func (t *g2048) Print() {
	for _, line := range t {
		for _, number := range line {
			fmt.Printf("%2d ", number)
		}
		fmt.Println()
	}
	fmt.Println()
	tn := g2048{{1, 2, 3, 4}, {5, 8}, {9, 10, 11}, {13, 14, 16}}
	*t = tn

}

func main() {
	fmt.Println("origin")
	t := g2048{{1, 2, 3, 4}, {5, 8}, {9, 10, 11}, {13, 14, 16}}
	t.Print()
	fmt.Println("mirror")
	t.MirrorV()
	t.Print()
	fmt.Println("Left90")
	t.Left90()
	t.Print()
	fmt.Println("Right90")
	t.R90()
	t.Print()
	fmt.Println("Right180")
	t.Right180()
	t.Print()
}

/*
输出

origin
 1  2  3  4
 5  8  0  0
 9 10 11  0
13 14 16  0

mirror
13 14 16  0
 9 10 11  0
 5  8  0  0
 1  2  3  4

Left90
 4  0  0  0
 3  0 11 16
 2  8 10 14
 1  5  9 13

Right90
13  9  5  1
14 10  8  2
16 11  0  3
 0  0  0  4

Right180
 0 16 14 13
 0 11 10  9
 0  0  8  5
 4  3  2  1

*/
