program SierpinskiCarpet
  implicit none
  integer :: L, iterations
  integer, parameter :: V0 = 10**8
  integer, dimension(:,:), allocatable :: potential
  integer :: i, Nrec

  read(5,*) iterations
  L=3**iterations
  allocate(potential(L,L))

  open(10, file="Sierpinski.dat")

  do Nrec = 1, int(L**(1./3.))
    call generateSierpinskiCarpet(L, Nrec, V0, potential)
  end do

  do i = 1,L
    write(10,*) potential(i,:)
  end do

  deallocate(potential)

  


contains


subroutine generateSierpinskiCarpet(L, Nrec, V0, potential)
  implicit none
  integer, intent(in) :: L          ! Size of the system
  integer, intent(in) :: Nrec       ! Recursion number
  integer, intent(in) :: V0         ! Potential strength
  integer, intent(out) :: potential(L, L) ! Output potential array
  integer :: ix, iy

  ! Initialize potential array
  potential = 0.0

  ! Generate Sierpinski carpet
  ix = 1
  iy = 1
  call fillSierpinskiCarpet(ix, iy, L, Nrec, V0, potential)
  
end subroutine generateSierpinskiCarpet

recursive subroutine fillSierpinskiCarpet(ix, iy, size, Nrec, V0, potential)
  implicit none
  integer, intent(in) :: ix, iy    ! Coordinates of the top-left corner of the square
  integer, intent(in) :: size       ! Size of the square
  integer, intent(in) :: Nrec       ! Recursion number
  integer, intent(in) :: V0         ! Potential strength
  integer, intent(out) :: potential(:,:) ! Output potential array

  integer :: i, j, new_size

  ! Check recursion limit
  if (Nrec <= 0) return

  ! Calculate the new size of the smaller squares
  new_size = size / 3

  ! Fill the central square with potential V0
  potential(ix+new_size:ix+2*new_size-1, iy+new_size:iy+2*new_size-1) = V0

  ! Recursively fill the remaining eight smaller squares
  do i = 0, 2
    do j = 0, 2
      if (i == 1 .and. j == 1) then
        ! Skip the central square
        cycle
      else
        ! Recursive call for each smaller square
        call fillSierpinskiCarpet(ix + i * new_size, iy + j * new_size, new_size, Nrec - 1, V0, potential)
      end if
    end do
  end do

end subroutine fillSierpinskiCarpet

end program