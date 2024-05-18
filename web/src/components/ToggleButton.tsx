import { HTMLProps } from "react";
import { HamburgerMenu } from "../Icons/HamburgerMenu.tsx";

export function ToggleButton ({ className = '', ...rest }: HTMLProps<HTMLButtonElement>) {
  return <button
    {...rest}
    type={'button'}
    className={`bg-[var(--drawer-background-color)] text-[var(--text-color)] rounded-[50%] size-10 ${className}`}
  >
    <HamburgerMenu className={'m-auto'}/>
  </button>
}
